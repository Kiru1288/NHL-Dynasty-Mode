"""How a CPU front office judges a roster move: does it make tonight's lineup better?

Mirrors the game's team-strength model (engine._team_strength / _roster_depth_strength):
four forward lines, three pairs and the crease all count, a starting goalie is the
single biggest piece, a thin bottom six / third pair is punished, and the best-12
average only counts for 15%. Side-effect free: works on a list of players, never
touches the real roster or line assignments.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from app.sim_engine.trades.team_assessment import player_ovr, position_group

LINE_W = (0.29, 0.26, 0.24, 0.21)
PAIR_W = (0.36, 0.33, 0.31)
EMPTY = 60.0  # a missing body is a call-up


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else EMPTY


def _ovr(p: Any) -> float:
    v = player_ovr(p)
    if v <= 0:
        try:
            raw = float(getattr(p, "overall", 0) or 0)
        except (TypeError, ValueError):
            raw = 0.0
        v = raw * 99.0 if 0 < raw <= 1.5 else raw
    return v if v > 0 else EMPTY


def _pid(p: Any) -> str:
    return str(getattr(p, "id", None) or getattr(p, "player_id", None) or id(p))


def lineup_score(players: Iterable[Any]) -> float:
    """0..1 team strength of the best lineup these players can dress."""
    fs: List[float] = []
    ds: List[float] = []
    gs: List[float] = []
    for p in players:
        if p is None or getattr(p, "retired", False):
            continue
        g = position_group(p)
        v = _ovr(p)
        if g == "G":
            gs.append(v)
        elif g in ("LD", "RD"):
            ds.append(v)
        else:
            fs.append(v)
    fs = sorted(fs, reverse=True)[:12]
    ds = sorted(ds, reverse=True)[:6]
    gs = sorted(gs, reverse=True)[:2]
    fs += [EMPTY] * (12 - len(fs))
    ds += [EMPTY] * (6 - len(ds))
    gs += [EMPTY] * (2 - len(gs))
    lines = [_mean(fs[i * 3 : i * 3 + 3]) / 99.0 for i in range(4)]
    pairs = [_mean(ds[i * 2 : i * 2 + 2]) / 99.0 for i in range(3)]
    f = sum(w * v for w, v in zip(LINE_W, lines))
    d = sum(w * v for w, v in zip(PAIR_W, pairs))
    g = (0.78 * gs[0] + 0.22 * gs[1]) / 99.0
    depth = (lines[2] + lines[3] + pairs[2]) / 3.0
    top = (lines[0] + pairs[0]) / 2.0
    dropoff = max(0.0, (top - depth) - 0.08)
    depth_s = 0.46 * f + 0.36 * d + 0.18 * g + (depth - 0.76) * 0.27 - dropoff * 0.15
    comp = _mean(sorted(fs + ds + gs, reverse=True)[:12]) / 99.0
    return 0.15 * comp + 0.85 * depth_s


def team_players(team: Any) -> List[Any]:
    """NHL roster plus anyone on IR (they come back)."""
    out = list(getattr(team, "roster", None) or [])
    seen = {_pid(p) for p in out}
    for p in list(getattr(team, "injured_reserve", None) or []):
        if _pid(p) not in seen:
            out.append(p)
    return out


def roster_delta(team: Any, add: Iterable[Any] = (), remove: Iterable[Any] = (), *, base: Optional[List[Any]] = None) -> float:
    """Change in lineup strength if ``add`` joins and ``remove`` leaves (≈ -0.05..+0.05).

    Rough scale: +0.01 is a real upgrade (a solid bottom-six forward over a call-up),
    +0.025 is a starting goalie over a backup."""
    players = list(base) if base is not None else team_players(team)
    out_ids = {_pid(p) for p in remove}
    before = lineup_score(players)
    after_players = [p for p in players if _pid(p) not in out_ids] + [p for p in add if p is not None]
    return lineup_score(after_players) - before
