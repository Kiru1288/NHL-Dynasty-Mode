"""CPU call-ups and send-downs: keep the best available lineup dressed.

Every few days each CPU club looks at its AHL affiliate. If a player on an NHL contract
there would make the NHL lineup better than its weakest regular at the same position
(judged with the same lineup model the game uses — trades/lineup_impact.py), the club
recalls him and sends the other player down:

* waiver-exempt players are simply assigned to the AHL;
* players who need waivers go on the wire (the user can claim them) — only for a
  clearer upgrade, and never a core / protected player;
* the swap has to fit under the cap and the 23-man roster.

Moves are logged in ``session.cpu_roster_moves_log`` (the social feed reports them).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

RUN_EVERY_DAYS = 4
MAX_MOVES_PER_TEAM = 2
MIN_GAIN_EXEMPT = 0.0012     # lineup-strength gain to bother with a simple swap
MIN_GAIN_WAIVERS = 0.0025    # bigger bar when the other guy has to clear waivers
LOG_KEEP = 300


def _tid(team: Any) -> str:
    raw = getattr(team, "team_id", None)
    if raw is None:
        raw = getattr(team, "id", None)
    return str(raw) if raw is not None else ""


def _bucket(p: Any) -> str:
    from app.sim_engine.trades.team_assessment import position_group  # noqa: WPS433

    g = position_group(p)
    return "G" if g == "G" else ("D" if g in ("LD", "RD") else "F")


def _healthy(p: Any) -> bool:
    from app.sim_engine.trades.team_assessment import _is_injured  # noqa: WPS433

    if getattr(p, "retired", False) or getattr(p, "on_waiver_wire", False):
        return False
    if str(getattr(p, "waiver_status", "") or "").lower() == "on_waivers":
        return False
    return not _is_injured(p)


def _cap_fits(team: Any, league: Any, up: Any, down: Any) -> bool:
    from app.sim_engine.economy.cap_engine import (  # noqa: WPS433
        buried_cap_hit_millions,
        calculate_team_cap_snapshot,
        player_cap_hit_millions,
    )

    try:
        snap = calculate_team_cap_snapshot(team, league=league)
        space = float(snap.get("usableCapSpace") or 0.0)
        freed = player_cap_hit_millions(down) - buried_cap_hit_millions(down)
        added = player_cap_hit_millions(up) - buried_cap_hit_millions(up)
        return space + freed - added >= -0.001
    except Exception:
        return False


def _send_down(team: Any, player: Any) -> None:
    team.roster = [p for p in list(getattr(team, "roster", None) or []) if p is not player]
    ahl = list(getattr(team, "ahl_roster", None) or [])
    if player not in ahl:
        ahl.append(player)
    team.ahl_roster = ahl
    for attr, val in (("in_minors", True), ("is_buried", False), ("buried", False), ("roster_location", "ahl")):
        try:
            setattr(player, attr, val)
        except Exception:
            pass


def _call_up(team: Any, player: Any) -> None:
    team.ahl_roster = [p for p in list(getattr(team, "ahl_roster", None) or []) if p is not player]
    roster = list(getattr(team, "roster", None) or [])
    if player not in roster:
        roster.append(player)
    team.roster = roster
    for attr, val in (("in_minors", False), ("is_buried", False), ("buried", False), ("roster_location", "nhl"), ("waiver_status", None)):
        try:
            setattr(player, attr, val)
        except Exception:
            pass


def _log_move(session: Any, team: Any, up: Any, down: Optional[Any], method: str, gain: float) -> None:
    from services.contract_economy import _player_id, _player_name, _player_ovr  # noqa: WPS433

    day = int(getattr(session, "calendar_cursor", 0) or 0)
    row = {
        "id": f"crm_{day}_{_tid(team)}_{_player_id(up)}",
        "day": day,
        "season": int(getattr(session, "season_calendar_year", 0) or 0),
        "team_id": _tid(team),
        "abbr": str(getattr(team, "abbreviation", None) or getattr(team, "abbr", "") or "").upper(),
        "up": {"player_id": _player_id(up), "name": _player_name(up), "ovr": round(_player_ovr(up)), "age": int(getattr(getattr(up, "identity", None), "age", 0) or 0)},
        "down": ({"player_id": _player_id(down), "name": _player_name(down), "ovr": round(_player_ovr(down))} if down is not None else None),
        "summary": "",
        "method": method,
        "gain": round(gain, 4),
    }
    log = list(getattr(session, "cpu_roster_moves_log", None) or [])
    log.append(row)
    session.cpu_roster_moves_log = log[-LOG_KEEP:]


def _fix_composition(session: Any, team: Any, league: Any) -> int:
    """Carry two healthy goalies (not four) and dress twelve forwards and six D."""
    from app.sim_engine.economy.cap_engine import can_recall_player  # noqa: WPS433
    from app.sim_engine.trades.trade_asset import player_holds_nhl_spc  # noqa: WPS433
    from services.contract_economy import _player_ovr, is_core_player_protected, is_waiver_exempt  # noqa: WPS433

    moves = 0
    healthy = [p for p in list(getattr(team, "roster", None) or []) if _healthy(p)]
    goalies = sorted([p for p in healthy if _bucket(p) == "G"], key=_player_ovr, reverse=True)
    short_skaters = (len([p for p in healthy if _bucket(p) == "F"]) < 12 or len([p for p in healthy if _bucket(p) == "D"]) < 6)
    # A third goalie is fine; a fourth (or a third while dressing short) wastes a spot.
    for g in goalies[(2 if short_skaters else 3):]:
        if is_waiver_exempt(g, team, league):
            _send_down(team, g)
            _log_move(session, team, g, None, "goalie_assigned", 0.0)
            moves += 1
        elif _player_ovr(g) < 80:
            try:
                if is_core_player_protected(g, team, league):
                    continue
            except Exception:
                continue
            from services.waivers import place_on_waivers  # noqa: WPS433

            if place_on_waivers(session, team, g, reason="cpu_roster_move", manual=False).get("ok"):
                _log_move(session, team, g, None, "goalie_waived", 0.0)
                moves += 1
    for bucket, need in (("F", 12), ("D", 6), ("G", 2)):
        for _ in range(3):
            roster = list(getattr(team, "roster", None) or [])
            have = len([p for p in roster if _healthy(p) and _bucket(p) == bucket])
            if have >= need or len(roster) >= 23:
                break
            pool = sorted([p for p in list(getattr(team, "ahl_roster", None) or []) if _healthy(p) and _bucket(p) == bucket and player_holds_nhl_spc(p)],
                          key=_player_ovr, reverse=True)
            up = next((p for p in pool if can_recall_player(team, p, league).get("ok")), None)
            if up is None:
                break
            _call_up(team, up)
            _log_move(session, team, up, None, "recalled", 0.0)
            moves += 1
    return moves


def _team_moves(session: Any, team: Any, league: Any) -> int:
    from app.sim_engine.trades.lineup_impact import roster_delta, team_players  # noqa: WPS433
    from app.sim_engine.trades.trade_asset import player_holds_nhl_spc  # noqa: WPS433
    from services.contract_economy import _player_ovr, is_core_player_protected, is_waiver_exempt, sync_team_cap_fields  # noqa: WPS433

    moves = _fix_composition(session, team, league)
    tried: set = set()
    while moves < MAX_MOVES_PER_TEAM + 2:
        nhl = [p for p in list(getattr(team, "roster", None) or []) if _healthy(p)]
        ahl = [p for p in list(getattr(team, "ahl_roster", None) or []) if _healthy(p) and player_holds_nhl_spc(p) and id(p) not in tried]
        if not ahl or not nhl:
            return moves
        best: Optional[tuple] = None
        base = team_players(team)
        for up in sorted(ahl, key=_player_ovr, reverse=True)[:8]:
            b = _bucket(up)
            same = sorted([p for p in nhl if _bucket(p) == b], key=_player_ovr)
            if not same:
                continue
            # Weakest few at the position are the candidates to go down.
            for down in same[:3]:
                if _player_ovr(up) <= _player_ovr(down):
                    continue
                exempt = is_waiver_exempt(down, team, league)
                if not exempt:
                    try:
                        if is_core_player_protected(down, team, league) or _player_ovr(down) >= 80:
                            continue
                    except Exception:
                        continue
                gain = roster_delta(team, add=[up], remove=[down], base=base)
                bar = MIN_GAIN_EXEMPT if exempt else MIN_GAIN_WAIVERS
                if gain < bar or not _cap_fits(team, league, up, down):
                    continue
                if best is None or gain > best[0]:
                    best = (gain, up, down, exempt)
        if best is None:
            return moves
        gain, up, down, exempt = best
        tried.add(id(up))
        if exempt:
            _send_down(team, down)
            method = "assigned"
        else:
            from services.waivers import place_on_waivers  # noqa: WPS433

            res = place_on_waivers(session, team, down, reason="cpu_roster_move", manual=False)
            if not res.get("ok"):
                continue
            method = "waived"
        _call_up(team, up)
        try:
            sync_team_cap_fields(team, league)
        except Exception:
            _log.debug("cap sync failed", exc_info=True)
        _log_move(session, team, up, down, method, gain)
        moves += 1
    return moves


def run_cpu_roster_moves(session: Any, *, force: bool = False) -> Dict[str, Any]:
    """In-season, every few days: CPU clubs recall AHL players who'd improve the lineup."""
    phase = str(getattr(session, "phase", "") or "").lower()
    if phase not in ("regular", "playoffs") and not force:
        return {"moves": 0}
    day = int(getattr(session, "calendar_cursor", 0) or 0)
    last = int(getattr(session, "_cpu_roster_moves_day", -99) or -99)
    if not force and 0 <= day - last < RUN_EVERY_DAYS:
        return {"moves": 0}
    session._cpu_roster_moves_day = day
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return {"moves": 0}
    user_tid = str(getattr(session, "user_team_id", "") or "")
    total = 0
    # Weekly: clubs under the salary floor take on contracts capped-out clubs want gone.
    floor_trades = 0
    if force or day - int(getattr(session, "_cpu_cap_floor_day", -99) or -99) >= 7:
        session._cpu_cap_floor_day = day
        try:
            from services.contract_economy import run_cpu_cap_floor_pass  # noqa: WPS433

            floor_trades = int(run_cpu_cap_floor_pass(session).get("count") or 0)
        except Exception:
            _log.exception("cap floor pass failed")
    for team in list(getattr(league, "teams", None) or []):
        if _tid(team) == user_tid:
            continue
        try:
            total += _team_moves(session, team, league)
        except Exception:
            _log.exception("cpu roster moves failed for %s", _tid(team))
    return {"moves": total, "cap_floor_trades": floor_trades}
