from __future__ import annotations

"""
League standings and ranking system.

This module tracks team records over a season and exposes sorted standings
views for league-wide, conference, and division contexts. It is designed
to be agnostic to the concrete Team implementation; only a handful of
attributes are probed via getattr.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


POINTS_FOR_WIN = 2
POINTS_FOR_OTL = 1  # treated as overtime/shootout loss when flagged


def _safe_team_id(team: Any, fallback_index: int) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", None)
    if tid is None:
        return f"T{fallback_index:02d}"
    return str(tid)


def _safe_team_name(team: Any, team_id: str) -> str:
    name = getattr(team, "name", None)
    city = getattr(team, "city", None)
    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)
    return str(team_id)


def _safe_conf(team: Any) -> Optional[str]:
    return getattr(team, "conference", None)


def _safe_div(team: Any) -> Optional[str]:
    return getattr(team, "division", None)


@dataclass
class TeamStandingRecord:
    team_id: str
    name: str
    conference: Optional[str] = None
    division: Optional[str] = None

    gp: int = 0
    wins: int = 0
    losses: int = 0
    otl: int = 0
    points: int = 0

    gf: int = 0
    ga: int = 0
    rw: int = 0  # regulation wins (approximate: any non-OT win)

    last_10: List[str] = field(default_factory=list)  # "W" / "L" / "O"

    def goal_diff(self) -> int:
        return self.gf - self.ga

    def point_pct(self) -> float:
        if self.gp == 0:
            return 0.0
        return self.points / float(self.gp * POINTS_FOR_WIN)

    def streak(self) -> str:
        if not self.last_10:
            return "-"
        # compress most recent run as simple label, e.g. W3 / L2
        current = self.last_10[-1]
        count = 1
        for r in reversed(self.last_10[:-1]):
            if r == current:
                count += 1
            else:
                break
        return f"{current}{count}"


class StandingsTable:
    """
    Container for all team records and helpers to derive sorted views
    and playoff-qualification related queries.
    """

    def __init__(self, teams: Iterable[Any]):
        self.records: Dict[str, TeamStandingRecord] = {}
        self._by_conf: Dict[str, List[str]] = {}
        self._by_div: Dict[Tuple[str, str], List[str]] = {}
        for idx, t in enumerate(teams):
            tid = _safe_team_id(t, idx)
            name = _safe_team_name(t, tid)
            conf = _safe_conf(t)
            div = _safe_div(t)
            rec = TeamStandingRecord(team_id=tid, name=name, conference=conf, division=div)
            self.records[tid] = rec
            if conf:
                self._by_conf.setdefault(conf, []).append(tid)
            if conf and div:
                self._by_div.setdefault((conf, div), []).append(tid)

    # ------------------------------------------------------------------
    # Updating from games
    # ------------------------------------------------------------------

    def record_game(
        self,
        home_id: str,
        away_id: str,
        home_goals: int,
        away_goals: int,
        overtime: bool = False,
    ) -> None:
        """
        Update standings for a single completed game.

        Simplified NHL-like rules:
        - Win: 2 points
        - Overtime/shootout loss: 1 point
        - Regulation loss: 0 points
        """
        if home_id not in self.records or away_id not in self.records:
            return
        h = self.records[home_id]
        a = self.records[away_id]

        h.gp += 1
        a.gp += 1
        h.gf += home_goals
        h.ga += away_goals
        a.gf += away_goals
        a.ga += home_goals

        if home_goals > away_goals:
            # home win
            h.wins += 1
            h.points += POINTS_FOR_WIN
            if not overtime:
                h.rw += 1
            if overtime:
                a.otl += 1
                a.points += POINTS_FOR_OTL
            else:
                a.losses += 1
            h.last_10.append("W")
            a.last_10.append("O" if overtime else "L")
        elif away_goals > home_goals:
            # away win
            a.wins += 1
            a.points += POINTS_FOR_WIN
            if not overtime:
                a.rw += 1
            if overtime:
                h.otl += 1
                h.points += POINTS_FOR_OTL
            else:
                h.losses += 1
            a.last_10.append("W")
            h.last_10.append("O" if overtime else "L")
        else:
            # exact tie fallback: treat as OT win for random side
            # caller should avoid this by resolving ties, but we
            # keep a deterministic tiebreak by picking lexicographically.
            winner, loser = (h, a) if h.team_id <= a.team_id else (a, h)
            winner.wins += 1
            winner.points += POINTS_FOR_WIN
            loser.otl += 1
            loser.points += POINTS_FOR_OTL
            winner.rw += 1
            winner.last_10.append("W")
            loser.last_10.append("O")

        # Trim last_10 history
        if len(h.last_10) > 10:
            h.last_10 = h.last_10[-10:]
        if len(a.last_10) > 10:
            a.last_10 = a.last_10[-10:]

    # ------------------------------------------------------------------
    # Sorting / views
    # ------------------------------------------------------------------

    def _sort_key(self, rec: TeamStandingRecord) -> Tuple:
        # Higher first: points, wins, regulation wins, goal difference, goals for.
        # Deterministic fallback: team_id lexicographically.
        return (
            rec.points,
            rec.wins,
            rec.rw,
            rec.goal_diff(),
            rec.gf,
            rec.team_id * -1,  # not actually used; placeholder
        )

    def league_table(self) -> List[TeamStandingRecord]:
        """Full league standings sorted best -> worst."""
        return sorted(self.records.values(), key=self._sort_key, reverse=True)

    def conference_table(self, conference: str) -> List[TeamStandingRecord]:
        tids = self._by_conf.get(conference, [])
        recs = [self.records[tid] for tid in tids]
        return sorted(recs, key=self._sort_key, reverse=True)

    def division_table(self, conference: str, division: str) -> List[TeamStandingRecord]:
        tids = self._by_div.get((conference, division), [])
        recs = [self.records[tid] for tid in tids]
        return sorted(recs, key=self._sort_key, reverse=True)

    # Convenience helpers ------------------------------------------------

    def top_n(self, n: int) -> List[TeamStandingRecord]:
        return self.league_table()[: max(0, n)]

    def bottom_n(self, n: int) -> List[TeamStandingRecord]:
        tbl = self.league_table()
        return list(reversed(tbl))[: max(0, n)]

    def playoff_seeds_by_conference(self, per_conf: int = 8) -> Dict[str, List[TeamStandingRecord]]:
        """
        Return top N seeds by conference. If no conference information
        exists, the dict contains a single pseudo-conference "ALL".
        """
        if not self._by_conf:
            tbl = self.league_table()[:per_conf]
            return {"ALL": tbl}
        out: Dict[str, List[TeamStandingRecord]] = {}
        for conf in sorted(self._by_conf.keys()):
            out[conf] = self.conference_table(conf)[:per_conf]
        return out

    def presidents_trophy_winner(self) -> Optional[TeamStandingRecord]:
        tbl = self.league_table()
        return tbl[0] if tbl else None

    def as_table_rows(self) -> List[Dict[str, Any]]:
        """Return a plain-JSON-friendly representation of standings."""
        rows: List[Dict[str, Any]] = []
        for rec in self.league_table():
            rows.append(
                {
                    "team_id": rec.team_id,
                    "name": rec.name,
                    "conference": rec.conference,
                    "division": rec.division,
                    "gp": rec.gp,
                    "wins": rec.wins,
                    "losses": rec.losses,
                    "otl": rec.otl,
                    "points": rec.points,
                    "gf": rec.gf,
                    "ga": rec.ga,
                    "goal_diff": rec.goal_diff(),
                    "rw": rec.rw,
                    "point_pct": rec.point_pct(),
                    "streak": rec.streak(),
                }
            )
        return rows


