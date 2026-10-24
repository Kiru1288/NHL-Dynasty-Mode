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


def _call_up_from_pool(team: Any, player: Any) -> None:
    """Junior/European prospect on a signed ELC jumping straight to the NHL."""
    team.prospect_pool = [p for p in list(getattr(team, "prospect_pool", None) or []) if p is not player]
    _call_up(team, player)


def _season_jump(p: Any) -> float:
    """Rating points gained since opening night (in-season development)."""
    from services.contract_economy import _player_ovr  # noqa: WPS433

    start = getattr(p, "season_start_ovr", None) or getattr(p, "_season_start_ovr", None)
    try:
        start_f = float(start)
    except (TypeError, ValueError):
        return 0.0
    if start_f <= 1.5:
        start_f *= 99.0
    return max(0.0, _player_ovr(p) - start_f)


def _ahl_form(session: Any, p: Any) -> tuple:
    """(rating points of credit, reason) for AHL production vs. what his rating predicts."""
    from services.contract_economy import _player_id, _player_ovr  # noqa: WPS433

    try:
        from services.ahl_league import get_player_ahl_season_line  # noqa: WPS433

        line = get_player_ahl_season_line(session, _player_id(p))
    except Exception:
        line = None
    if not line:
        return 0.0, ""
    gp = int(line.get("gp") or 0)
    if gp < 10:
        return 0.0, ""
    pts = int(line.get("points") or 0)
    ppg = pts / gp
    is_d = _bucket(p) == "D"
    exp = max(0.15, 0.25 + 0.04 * (_player_ovr(p) - 65.0)) * (0.5 if is_d else 1.0)
    credit = max(-2.0, min(4.0, (ppg - exp) * (12.0 if is_d else 8.0))) * min(1.0, gp / 20.0)
    reason = f"{pts} pts in {gp} AHL games" if credit > 0.5 else ""
    return credit, reason


def _nhl_form(session: Any, p: Any) -> float:
    """Rating points off (or on) for how an NHL regular is producing this season."""
    from services.contract_economy import _player_id, _player_ovr  # noqa: WPS433

    row = (getattr(session, "player_season_stats", None) or {}).get(_player_id(p))
    if not isinstance(row, dict):
        return 0.0
    gp = int(row.get("gp") or 0)
    if gp < 10:
        return 0.0
    ppg = (int(row.get("g") or 0) + int(row.get("a") or 0)) / gp
    is_d = _bucket(p) == "D"
    exp = max(0.05, min(1.4, 0.10 + 0.042 * (_player_ovr(p) - 68.0))) * (0.45 if is_d else 1.0)
    return max(-3.0, min(3.0, (ppg - exp) * (20.0 if is_d else 12.0))) * min(1.0, gp / 25.0)


def _prospect_pool_callup_eligible(session: Any, p: Any, nhl: List[Any], tried: set) -> bool:
    """Signed CHL/Europe pool players: big in-season jump or clear upgrade at a position."""
    from app.sim_engine.trades.trade_asset import player_holds_nhl_spc  # noqa: WPS433
    from services.contract_economy import _player_ovr  # noqa: WPS433

    if not (_healthy(p) and player_holds_nhl_spc(p) and id(p) not in tried):
        return False
    if _season_jump(p) >= 4.0:
        return True
    b = _bucket(p)
    same = [x for x in nhl if _bucket(x) == b]
    ovr = _player_ovr(p)
    if not same:
        return ovr >= (72 if b == "F" else 74)
    weakest = min(_value(session, x, nhl=True)[0] for x in same)
    up_val, _ = _value(session, p, nhl=False)
    margin = 1.0 if b == "F" else 1.5
    return up_val >= weakest + margin


def _value(session: Any, p: Any, *, nhl: bool) -> tuple:
    """(rating + form, reason) used to compare a call-up with the player he replaces."""
    from services.contract_economy import _player_ovr  # noqa: WPS433

    if nhl:
        return _player_ovr(p) + _nhl_form(session, p), ""
    credit, reason = _ahl_form(session, p)
    jump = _season_jump(p)
    if jump >= 4.0:
        credit += min(3.0, (jump - 3.0) * 0.75)
        reason = (reason + "; " if reason else "") + f"up {jump:.0f} OVR this season"
    return _player_ovr(p) + credit, reason


def _log_move(session: Any, team: Any, up: Any, down: Optional[Any], method: str, gain: float, reason: str = "") -> None:
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
        "summary": reason,
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
        juniors = [
            p for p in list(getattr(team, "prospect_pool", None) or [])
            if _prospect_pool_callup_eligible(session, p, nhl, tried)
        ]
        if not (ahl or juniors) or not nhl:
            return moves
        best: Optional[tuple] = None
        base = team_players(team)
        # Rating + form: AHL production and in-season jumps count, and so does a
        # slumping NHL regular. Ratings alone never rewarded a hot farmhand.
        cand = sorted(ahl + juniors, key=lambda p: _value(session, p, nhl=False)[0], reverse=True)[:10]
        for up in cand:
            b = _bucket(up)
            up_val, why = _value(session, up, nhl=False)
            same = sorted([p for p in nhl if _bucket(p) == b], key=lambda p: _value(session, p, nhl=True)[0])
            if not same:
                continue
            # Weakest few at the position (rating + form) are the candidates to go down.
            for down in same[:3]:
                down_val = _value(session, down, nhl=True)[0]
                form_driven = bool(why) and _player_ovr(up) <= _player_ovr(down)
                if up_val <= down_val + (1.0 if form_driven else 0.0):
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
                if form_driven:
                    # The lineup model only sees ratings; a form call-up only has to
                    # not hurt it much, and never costs a waiver claim.
                    if not exempt:
                        continue
                    bar = -0.003
                if gain < bar or not _cap_fits(team, league, up, down):
                    continue
                score = gain + 0.001 * (up_val - down_val)
                if best is None or score > best[0]:
                    best = (score, up, down, exempt, why, gain)
        if best is None:
            return moves
        _score, up, down, exempt, why, gain = best
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
        if any(p is up for p in list(getattr(team, "prospect_pool", None) or [])):
            _call_up_from_pool(team, up)
        else:
            _call_up(team, up)
        try:
            sync_team_cap_fields(team, league)
        except Exception:
            _log.debug("cap sync failed", exc_info=True)
        _log_move(session, team, up, down, method, gain, why)
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
    _floor_gap = day - int(getattr(session, "_cpu_cap_floor_day", -99) or -99)
    if force or _floor_gap >= 7 or _floor_gap < 0:  # < 0: last season's stamp
        session._cpu_cap_floor_day = day
        try:
            from services.contract_economy import run_cpu_cap_floor_pass  # noqa: WPS433

            floor_trades = int(run_cpu_cap_floor_pass(session).get("count") or 0)
        except Exception:
            _log.exception("cap floor pass failed")
        # Clubs with room sell it to cap-strapped contenders for picks, up to the deadline.
        try:
            from services.contract_economy import run_cpu_cap_space_broker_pass  # noqa: WPS433
            from services.franchise_sim import _calendar_trade_deadline_index  # noqa: WPS433

            deadline = _calendar_trade_deadline_index(session)
            if phase == "regular" and (deadline is None or day <= int(deadline)):
                floor_trades += int(run_cpu_cap_space_broker_pass(session).get("count") or 0)
        except Exception:
            _log.exception("cap space broker pass failed")
    for team in list(getattr(league, "teams", None) or []):
        if _tid(team) == user_tid:
            continue
        try:
            total += _team_moves(session, team, league)
        except Exception:
            _log.exception("cpu roster moves failed for %s", _tid(team))
    return {"moves": total, "cap_floor_trades": floor_trades}
