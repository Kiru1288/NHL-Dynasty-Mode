"""
Post-trade roster balancing ("corresponding moves").

Real clubs never fail a trade over the 23-man limit; they make the deal and send
someone to the AHL the same day. Validation uses ``demotion_capacity`` to know how
many bodies a club can shed, and the executor calls ``auto_send_down_overflow`` after
the players have moved.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

ACTIVE_MAX = 23
MAX_AUTO_SEND_DOWNS = 3
_POS_MIN = {"F": 12, "D": 6, "G": 2}


def _pid(p: Any) -> str:
    return str(getattr(p, "id", "") or "")


def _pos_bucket(p: Any) -> str:
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident is not None else getattr(p, "position", "")
    pos = str(getattr(pos, "value", pos) or "").upper()
    if pos.startswith("G"):
        return "G"
    if pos.startswith("D"):
        return "D"
    return "F"


def _ovr(p: Any) -> float:
    fn = getattr(p, "ovr", None)
    try:
        v = float(fn()) if callable(fn) else float(fn or getattr(p, "overall", 0) or 0)
    except Exception:
        v = 0.0
    return v * 99.0 if v <= 1.5 else v


def _is_active(p: Any) -> bool:
    try:
        from app.sim_engine.economy.cap_engine import _is_active_roster_player

        return bool(_is_active_roster_player(p))
    except Exception:
        return not bool(getattr(p, "in_minors", False))


def _has_nmc(p: Any) -> bool:
    c = getattr(p, "contract", None)
    if isinstance(c, dict):
        return bool(c.get("no_move_clause") or c.get("nmc") or c.get("no_movement_clause"))
    if c is None:
        return False
    clauses = getattr(c, "clauses", None)
    if clauses is not None and getattr(clauses, "noMoveClause", False):
        return True
    return bool(getattr(c, "no_move_clause", False) or getattr(c, "nmc", False))


def _waiver_exempt(p: Any) -> bool:
    for attr in ("waiver_exempt", "is_waiver_exempt"):
        v = getattr(p, attr, None)
        if v is not None:
            return bool(v)
    ident = getattr(p, "identity", None)
    age = getattr(ident, "age", None) if ident is not None else getattr(p, "age", None)
    try:
        return int(age or 99) <= 21
    except Exception:
        return False


def _send_down_order(players: List[Any]) -> List[Any]:
    """Lowest-rated first; waiver-exempt players get a nudge since sending them is free."""
    return sorted(players, key=lambda p: _ovr(p) - (4.0 if _waiver_exempt(p) else 0.0))


def send_down_candidates(
    team: Any,
    *,
    leaving_ids: Iterable[str] = (),
    arriving: Iterable[Any] = (),
    protect_ids: Iterable[str] = (),
    count: int = MAX_AUTO_SEND_DOWNS,
) -> List[Any]:
    """Who the club would assign to the AHL, keeping 12F / 6D / 2G dressed."""
    leaving = {str(x) for x in leaving_ids}
    active = [p for p in list(getattr(team, "roster", None) or []) if _is_active(p) and _pid(p) not in leaving]
    arriving_ids = {_pid(p) for p in arriving if p is not None}
    protect = {str(x) for x in protect_ids}
    already = {_pid(p) for p in active}
    pool = active + [p for p in arriving if p is not None and _pid(p) not in already]
    by_pos: Dict[str, int] = {"F": 0, "D": 0, "G": 0}
    for p in pool:
        by_pos[_pos_bucket(p)] += 1
    movable = [
        p for p in active
        if _pid(p) not in arriving_ids and _pid(p) not in protect and not _has_nmc(p)
    ]
    out: List[Any] = []
    for p in _send_down_order(movable):
        if len(out) >= count:
            break
        b = _pos_bucket(p)
        if by_pos[b] - 1 < _POS_MIN[b]:
            continue
        by_pos[b] -= 1
        out.append(p)
    return out


def demotion_capacity(team: Any, *, leaving_ids: Iterable[str] = (), arriving: Iterable[Any] = ()) -> int:
    return len(send_down_candidates(team, leaving_ids=leaving_ids, arriving=arriving))


def _assign_ahl(team: Any, p: Any) -> None:
    team.roster = [x for x in list(getattr(team, "roster", None) or []) if x is not p]
    ahl = list(getattr(team, "ahl_roster", None) or [])
    if not any(x is p for x in ahl):
        ahl.append(p)
    team.ahl_roster = ahl
    try:
        p.in_minors = True
        p.roster_location = "ahl"
    except Exception:
        pass
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    try:
        from app.sim_engine.league_hierarchy_bootstrap import _set_assignment, _team_label

        _set_assignment(p, org_nhl_team_id=tid, level="ahl", club=_team_label(team))
    except Exception:
        pass


def auto_send_down_overflow(team: Any, *, protect_ids: Iterable[str] = (), active_max: int = ACTIVE_MAX) -> List[Dict[str, Any]]:
    """Assign players to the AHL until the club is at or under the active maximum."""
    protect = {str(x) for x in protect_ids}
    active = [p for p in list(getattr(team, "roster", None) or []) if _is_active(p)]
    over = len(active) - int(active_max)
    if over <= 0:
        return []
    arriving = []
    cands = send_down_candidates(team, arriving=arriving, protect_ids=protect, count=over)
    moves: List[Dict[str, Any]] = []
    for p in cands:
        _assign_ahl(team, p)
        ident = getattr(p, "identity", None)
        moves.append({
            "player_id": _pid(p),
            "player_name": str(getattr(ident, "name", None) or getattr(p, "name", "") or "?"),
            "team_id": str(getattr(team, "team_id", None) or getattr(team, "id", "") or ""),
            "to_level": "ahl",
        })
    return moves
