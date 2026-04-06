"""
Roster management.

Responsibilities:
- keep roster size within [20, 23]
- promote prospects when they are NHL-ready
- cut/waive declining veterans when overflowing or under cap pressure
- ensure basic positional balance (enough D/G)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.economy.player_value import PlayerValue
from app.sim_engine.economy.team_needs import TeamNeeds


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _pos(player: Any) -> str:
    p = getattr(player, "position", "")
    s = str(getattr(p, "value", p)).upper()
    if s in ("LW", "RW", "W"):
        return "W"
    if s == "C":
        return "C"
    if s == "D":
        return "D"
    if s == "G":
        return "G"
    return s


def _ovr(player: Any) -> float:
    fn = getattr(player, "ovr", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            return 0.0
    return _safe_float(getattr(player, "ovr", None), 0.0)


@dataclass
class RosterManager:
    roster_min: int = 20
    roster_max: int = 23
    promotion_threshold: float = 0.62

    def __post_init__(self) -> None:
        self.value_model = PlayerValue()
        self.needs_model = TeamNeeds()

    def manage(self, team: Any, league: Any) -> List[str]:
        """
        Returns list of log strings ("WAIVERS:", "PROMOTION:", etc.).
        Does not print; engine/runner handles printing.
        """
        logs: List[str] = []
        roster: List[Any] = list(getattr(team, "roster", None) or [])
        prospects: List[Any] = list(getattr(team, "prospects", None) or [])

        # Ensure team has storage
        try:
            team.roster = roster
            team.prospects = prospects
        except Exception:
            pass

        # Evaluate needs (used as a nudge for positional balance)
        needs = self.needs_model.evaluate(team)

        # Promote NHL-ready prospects if roster has room
        promoted = 0
        while len(roster) < self.roster_max:
            best = None
            best_score = self.promotion_threshold - 1.0
            for p in prospects:
                ov = _ovr(p)
                if ov >= self.promotion_threshold and ov > best_score:
                    best_score = ov
                    best = p
            if best is None:
                break
            prospects.remove(best)
            roster.append(best)
            promoted += 1
            logs.append(f"PROMOTION: {getattr(team, 'team_id', '?')} promotes {getattr(best, 'name', 'Prospect')} (OVR {best_score:.3f})")

        # If still short, fill from free agents (league.free_agents if exists)
        if len(roster) < self.roster_min:
            pool = list(getattr(league, "free_agents", None) or [])
            # sort by value; prefer younger if team is rebuilding
            arche = str(getattr(team, "status", getattr(team, "archetype", "")) or "").lower()
            rebuilding = ("rebuild" in arche) or ("tank" in arche)
            pool_sorted = sorted(
                pool,
                key=lambda p: (
                    self.value_model.evaluate(p, team=team),
                    -_safe_int(getattr(p, "age", 25), 25) if rebuilding else 0,
                    _ovr(p),
                ),
                reverse=True,
            )
            while len(roster) < self.roster_min and pool_sorted:
                p = pool_sorted.pop(0)
                if p in pool:
                    pool.remove(p)
                roster.append(p)
                try:
                    league.free_agents = pool
                except Exception:
                    pass
                logs.append(f"SIGNING: {getattr(team, 'team_id', '?')} signs {getattr(p, 'name', 'UFA')} (fill roster)")

        # If overflow, waive/cut lowest value players (prefer older/declining)
        if len(roster) > self.roster_max:
            # Keep minimal positional balance first: ensure at least 2 G and 6 D
            def protect(p: Any) -> float:
                pos = _pos(p)
                ov = _ovr(p)
                age = _safe_int(getattr(p, "age", 25), 25)
                val = self.value_model.evaluate(p, team=team)
                prot = val + ov * 0.10
                if pos == "G":
                    prot += 0.10
                if pos == "D":
                    prot += 0.05
                if age >= 34:
                    prot -= 0.06
                return prot

            roster_sorted = sorted(roster, key=protect)
            while len(roster) > self.roster_max and roster_sorted:
                cut = roster_sorted.pop(0)
                if cut not in roster:
                    continue
                roster.remove(cut)
                # send to waivers wire if league supports it
                try:
                    if not hasattr(league, "waiver_wire") or league.waiver_wire is None:
                        league.waiver_wire = []
                    league.waiver_wire.append({"player": cut, "from_team": getattr(team, "team_id", "?")})
                    logs.append(f"WAIVERS: {getattr(team, 'team_id', '?')} places {getattr(cut, 'name', 'Player')} on waivers")
                except Exception:
                    logs.append(f"CUT: {getattr(team, 'team_id', '?')} releases {getattr(cut, 'name', 'Player')}")

        # Store back
        try:
            team.roster = roster
            team.prospects = prospects
        except Exception:
            pass

        # Attach needs snapshot for other systems
        try:
            team.needs = needs
        except Exception:
            pass
        return logs


_DEFAULT = RosterManager()


def manage_roster(team: Any, league: Any) -> List[str]:
    return _DEFAULT.manage(team, league)

