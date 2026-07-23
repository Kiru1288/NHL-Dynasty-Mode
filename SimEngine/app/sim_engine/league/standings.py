from __future__ import annotations

"""
League standings and ranking system.

This module tracks team records over a season and exposes sorted standings
views for league-wide, conference, division, wildcard, and playoff-picture
contexts.

Designed for a hockey executive/franchise mode sim.

Major goals:
- More NHL-like standings logic
- Better frontend payloads
- League rank, conference rank, division rank, wildcard rank
- Playoff race labels
- Bubble/elimination/clinching-style context
- Safer math and cleaner formatting
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


POINTS_FOR_WIN = 2
POINTS_FOR_OTL = 1

DEFAULT_SEASON_GAMES = 82
DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE = 8
DEFAULT_DIVISION_AUTO_QUALIFIERS = 3
DEFAULT_WILDCARDS_PER_CONFERENCE = 2


# ----------------------------------------------------------------------
# Safe helpers
# ----------------------------------------------------------------------


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_team_id(team: Any, fallback_index: int) -> str:
    tid = getattr(team, "team_id", None)
    if tid is None:
        tid = getattr(team, "id", None)
    if tid is None:
        tid = getattr(team, "abbr", None)
    if tid is None:
        tid = getattr(team, "abbreviation", None)
    if tid is None:
        return f"T{fallback_index:02d}"
    return str(tid)


def _safe_team_abbr(team: Any, team_id: str) -> str:
    abbr = getattr(team, "abbr", None)
    if abbr is None:
        abbr = getattr(team, "abbreviation", None)
    if abbr:
        return str(abbr).upper()
    return str(team_id).upper()


def _safe_team_name(team: Any, team_id: str) -> str:
    name = getattr(team, "name", None)
    city = getattr(team, "city", None)

    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)

    nickname = getattr(team, "nickname", None)
    if city and nickname:
        return f"{city} {nickname}"
    if nickname:
        return str(nickname)

    return str(team_id)


def _safe_conf(team: Any) -> Optional[str]:
    conf = getattr(team, "conference", None)
    if conf is None:
        conf = getattr(team, "conf", None)
    return str(conf) if conf else None


def _safe_div(team: Any) -> Optional[str]:
    div = getattr(team, "division", None)
    if div is None:
        div = getattr(team, "div", None)
    return str(div) if div else None


def _pct(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _per_game(value: int, gp: int, digits: int = 2) -> float:
    if gp <= 0:
        return 0.0
    return round(float(value) / float(gp), digits)


def _signed_int(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def _record_string(wins: int, losses: int, otl: int) -> str:
    return f"{wins}-{losses}-{otl}"


def _last_10_string(results: List[str]) -> str:
    if not results:
        return "0-0-0"

    recent = results[-10:]
    wins = sum(1 for r in recent if r == "W")
    losses = sum(1 for r in recent if r == "L")
    otl = sum(1 for r in recent if r == "O")
    return f"{wins}-{losses}-{otl}"


def _streak_string(results: List[str]) -> str:
    if not results:
        return "-"

    current = results[-1]
    count = 1

    for r in reversed(results[:-1]):
        if r == current:
            count += 1
        else:
            break

    return f"{current}{count}"


def _form_label(results: List[str]) -> str:
    if not results:
        return "No Form"

    recent = results[-10:]
    wins = sum(1 for r in recent if r == "W")
    otl = sum(1 for r in recent if r == "O")
    points = wins * 2 + otl
    max_points = len(recent) * 2
    pct = points / max_points if max_points else 0.0

    if len(recent) < 5:
        return "Small Sample"
    if pct >= 0.750:
        return "Red Hot"
    if pct >= 0.625:
        return "Hot"
    if pct >= 0.500:
        return "Steady"
    if pct >= 0.375:
        return "Slipping"
    return "Ice Cold"


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class TeamStandingRecord:
    team_id: str
    name: str
    abbr: str = ""
    conference: Optional[str] = None
    division: Optional[str] = None

    gp: int = 0
    wins: int = 0
    losses: int = 0
    otl: int = 0
    points: int = 0

    gf: int = 0
    ga: int = 0

    # NHL-style tracking:
    # RW = regulation wins
    # ROW = regulation + overtime wins
    # SOW = shootout wins, included for frontend compatibility even if sim does not use shootouts
    rw: int = 0
    row: int = 0
    sow: int = 0

    home_gp: int = 0
    home_wins: int = 0
    home_losses: int = 0
    home_otl: int = 0

    away_gp: int = 0
    away_wins: int = 0
    away_losses: int = 0
    away_otl: int = 0

    last_10: List[str] = field(default_factory=list)

    def normalize(self) -> None:
        self.gp = max(0, _safe_int(self.gp))
        self.wins = max(0, _safe_int(self.wins))
        self.losses = max(0, _safe_int(self.losses))
        self.otl = max(0, _safe_int(self.otl))
        self.points = max(0, _safe_int(self.points))

        self.gf = max(0, _safe_int(self.gf))
        self.ga = max(0, _safe_int(self.ga))

        self.rw = max(0, _safe_int(self.rw))
        self.row = max(0, _safe_int(self.row))
        self.sow = max(0, _safe_int(self.sow))

        self.home_gp = max(0, _safe_int(self.home_gp))
        self.home_wins = max(0, _safe_int(self.home_wins))
        self.home_losses = max(0, _safe_int(self.home_losses))
        self.home_otl = max(0, _safe_int(self.home_otl))

        self.away_gp = max(0, _safe_int(self.away_gp))
        self.away_wins = max(0, _safe_int(self.away_wins))
        self.away_losses = max(0, _safe_int(self.away_losses))
        self.away_otl = max(0, _safe_int(self.away_otl))

        if len(self.last_10) > 10:
            self.last_10 = self.last_10[-10:]

    def goal_diff(self) -> int:
        return self.gf - self.ga

    def point_pct(self) -> float:
        if self.gp <= 0:
            return 0.0
        return self.points / float(self.gp * POINTS_FOR_WIN)

    def points_per_game(self) -> float:
        if self.gp <= 0:
            return 0.0
        return self.points / float(self.gp)

    def goals_for_per_game(self) -> float:
        return _per_game(self.gf, self.gp)

    def goals_against_per_game(self) -> float:
        return _per_game(self.ga, self.gp)

    def record(self) -> str:
        return _record_string(self.wins, self.losses, self.otl)

    def home_record(self) -> str:
        return _record_string(self.home_wins, self.home_losses, self.home_otl)

    def away_record(self) -> str:
        return _record_string(self.away_wins, self.away_losses, self.away_otl)

    def last_10_record(self) -> str:
        return _last_10_string(self.last_10)

    def streak(self) -> str:
        return _streak_string(self.last_10)

    def form_label(self) -> str:
        return _form_label(self.last_10)

    def games_remaining(self, season_games: int = DEFAULT_SEASON_GAMES) -> int:
        return max(0, season_games - self.gp)

    def max_possible_points(self, season_games: int = DEFAULT_SEASON_GAMES) -> int:
        return self.points + self.games_remaining(season_games) * POINTS_FOR_WIN

    def to_base_row(self, season_games: int = DEFAULT_SEASON_GAMES) -> Dict[str, Any]:
        self.normalize()

        return {
            "team_id": self.team_id,
            "name": self.name,
            "abbr": self.abbr or self.team_id,
            "conference": self.conference,
            "division": self.division,

            "gp": self.gp,
            "wins": self.wins,
            "losses": self.losses,
            "otl": self.otl,
            "record": self.record(),
            "points": self.points,

            "gf": self.gf,
            "ga": self.ga,
            "goal_diff": self.goal_diff(),
            "goal_diff_label": _signed_int(self.goal_diff()),

            "rw": self.rw,
            "row": self.row,
            "sow": self.sow,

            "point_pct": _pct(self.point_pct(), 3),
            "points_per_game": _pct(self.points_per_game(), 2),
            "goals_for_per_game": self.goals_for_per_game(),
            "goals_against_per_game": self.goals_against_per_game(),

            "home_gp": self.home_gp,
            "home_record": self.home_record(),
            "home_wins": self.home_wins,
            "home_losses": self.home_losses,
            "home_otl": self.home_otl,

            "away_gp": self.away_gp,
            "away_record": self.away_record(),
            "away_wins": self.away_wins,
            "away_losses": self.away_losses,
            "away_otl": self.away_otl,

            "last_10": self.last_10_record(),
            "last_10_raw": list(self.last_10[-10:]),
            "streak": self.streak(),
            "form": self.form_label(),

            "games_remaining": self.games_remaining(season_games),
            "max_possible_points": self.max_possible_points(season_games),
        }


# ----------------------------------------------------------------------
# Standings table
# ----------------------------------------------------------------------


class StandingsTable:
    """
    Container for all team records and helpers to derive sorted views,
    playoff fields, wildcard races, and frontend-ready standings payloads.
    """

    def __init__(self, teams: Iterable[Any], season_games: int = DEFAULT_SEASON_GAMES):
        self.season_games = max(1, _safe_int(season_games, DEFAULT_SEASON_GAMES))

        self.records: Dict[str, TeamStandingRecord] = {}
        self._by_conf: Dict[str, List[str]] = {}
        self._by_div: Dict[Tuple[str, str], List[str]] = {}

        for idx, t in enumerate(teams):
            tid = _safe_team_id(t, idx)
            abbr = _safe_team_abbr(t, tid)
            name = _safe_team_name(t, tid)
            conf = _safe_conf(t)
            div = _safe_div(t)

            rec = TeamStandingRecord(
                team_id=tid,
                name=name,
                abbr=abbr,
                conference=conf,
                division=div,
            )

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
        shootout: bool = False,
        stats_home_goals: Optional[int] = None,
        stats_away_goals: Optional[int] = None,
    ) -> None:
        """
        Update standings for a single completed game.

        NHL-like rules:
        - Win: 2 points
        - Overtime/shootout loss: 1 point
        - Regulation loss: 0 points

        Notes:
        - RW increments only for regulation wins.
        - ROW increments for regulation + overtime wins.
        - SOW increments for shootout wins.
        - If your sim does not separate OT from SO, leave shootout=False.
        """

        if home_id not in self.records or away_id not in self.records:
            return

        h = self.records[home_id]
        a = self.records[away_id]

        home_goals = max(0, _safe_int(home_goals))
        away_goals = max(0, _safe_int(away_goals))
        stat_hg = home_goals if stats_home_goals is None else max(0, _safe_int(stats_home_goals))
        stat_ag = away_goals if stats_away_goals is None else max(0, _safe_int(stats_away_goals))

        h.gp += 1
        a.gp += 1

        h.home_gp += 1
        a.away_gp += 1

        h.gf += stat_hg
        h.ga += stat_ag

        a.gf += stat_ag
        a.ga += stat_hg

        # Caller should avoid unresolved ties.
        # If tie somehow arrives, force deterministic OT-style result.
        if home_goals == away_goals:
            overtime = True
            shootout = False
            if h.team_id <= a.team_id:
                home_goals += 1
            else:
                away_goals += 1

        if home_goals > away_goals:
            self._apply_win_loss(
                winner=h,
                loser=a,
                winner_is_home=True,
                overtime=overtime,
                shootout=shootout,
            )
        else:
            self._apply_win_loss(
                winner=a,
                loser=h,
                winner_is_home=False,
                overtime=overtime,
                shootout=shootout,
            )

        h.normalize()
        a.normalize()

    def _apply_win_loss(
        self,
        winner: TeamStandingRecord,
        loser: TeamStandingRecord,
        winner_is_home: bool,
        overtime: bool,
        shootout: bool,
    ) -> None:
        winner.wins += 1
        winner.points += POINTS_FOR_WIN

        if winner_is_home:
            winner.home_wins += 1
        else:
            winner.away_wins += 1

        if overtime or shootout:
            loser.otl += 1
            loser.points += POINTS_FOR_OTL

            if winner_is_home:
                loser.away_otl += 1
            else:
                loser.home_otl += 1

            if shootout:
                winner.sow += 1
            else:
                winner.row += 1
        else:
            loser.losses += 1

            if winner_is_home:
                loser.away_losses += 1
            else:
                loser.home_losses += 1

            winner.rw += 1
            winner.row += 1

        winner.last_10.append("W")
        loser.last_10.append("O" if overtime or shootout else "L")

        winner.last_10 = winner.last_10[-10:]
        loser.last_10 = loser.last_10[-10:]

    # ------------------------------------------------------------------
    # Sorting / tiebreakers
    # ------------------------------------------------------------------

    def _sort_key(self, rec: TeamStandingRecord) -> Tuple:
        """
        NHL-ish tiebreaker order.

        Simplified order:
        1. Points
        2. Points percentage
        3. Regulation wins
        4. Regulation + overtime wins
        5. Total wins
        6. Goal differential
        7. Goals for
        8. Fewer goals against
        9. Stable team id

        Official NHL has more detail, including head-to-head and other
        specific procedures, but this is much closer than basic PTS/ROW.
        """

        return (
            rec.points,
            rec.point_pct(),
            rec.rw,
            rec.row,
            rec.wins,
            rec.goal_diff(),
            rec.gf,
            -rec.ga,
            rec.team_id,
        )

    def league_table(self) -> List[TeamStandingRecord]:
        return sorted(self.records.values(), key=self._sort_key, reverse=True)

    def conference_table(self, conference: str) -> List[TeamStandingRecord]:
        tids = self._by_conf.get(conference, [])
        recs = [self.records[tid] for tid in tids if tid in self.records]
        return sorted(recs, key=self._sort_key, reverse=True)

    def division_table(self, conference: str, division: str) -> List[TeamStandingRecord]:
        tids = self._by_div.get((conference, division), [])
        recs = [self.records[tid] for tid in tids if tid in self.records]
        return sorted(recs, key=self._sort_key, reverse=True)

    def divisions_for_conference(self, conference: str) -> List[str]:
        return sorted({d for (c, d) in self._by_div.keys() if c == conference and d})

    def conferences(self) -> List[str]:
        return sorted(self._by_conf.keys())

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def top_n(self, n: int) -> List[TeamStandingRecord]:
        return self.league_table()[: max(0, n)]

    def bottom_n(self, n: int) -> List[TeamStandingRecord]:
        tbl = self.league_table()
        return list(reversed(tbl))[: max(0, n)]

    def presidents_trophy_winner(self) -> Optional[TeamStandingRecord]:
        tbl = self.league_table()
        return tbl[0] if tbl else None

    def find_record(self, team_id: str) -> Optional[TeamStandingRecord]:
        return self.records.get(str(team_id))

    def rank_maps(self) -> Dict[str, Dict[str, int]]:
        league_rank: Dict[str, int] = {}
        conference_rank: Dict[str, int] = {}
        division_rank: Dict[str, int] = {}

        for idx, rec in enumerate(self.league_table(), start=1):
            league_rank[rec.team_id] = idx

        for conf in self.conferences():
            for idx, rec in enumerate(self.conference_table(conf), start=1):
                conference_rank[rec.team_id] = idx

        for (conf, div), tids in self._by_div.items():
            tbl = self.division_table(conf, div)
            for idx, rec in enumerate(tbl, start=1):
                division_rank[rec.team_id] = idx

        return {
            "league": league_rank,
            "conference": conference_rank,
            "division": division_rank,
        }

    # ------------------------------------------------------------------
    # Playoff fields
    # ------------------------------------------------------------------

    def playoff_seeds_by_conference(
        self,
        per_conf: int = DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE,
    ) -> Dict[str, List[TeamStandingRecord]]:
        """
        Return top N seeds by conference.

        If no conference information exists, returns a single pseudo-conference.
        """

        if not self._by_conf:
            return {"ALL": self.league_table()[:per_conf]}

        out: Dict[str, List[TeamStandingRecord]] = {}

        for conf in self.conferences():
            out[conf] = self.conference_table(conf)[:per_conf]

        return out

    def nhl_conference_playoff_eight(
        self,
        conference: str,
        division_auto_qualifiers: int = DEFAULT_DIVISION_AUTO_QUALIFIERS,
        wildcards: int = DEFAULT_WILDCARDS_PER_CONFERENCE,
    ) -> Optional[List[TeamStandingRecord]]:
        """
        Build NHL-style 8-team playoff field for a conference:
        - Top 3 per division
        - Next 2 best records among remaining conference teams

        Returns None if the conference structure cannot support this.
        """

        divs = self.divisions_for_conference(conference)

        if len(divs) < 2:
            return None

        tids_conf = list(self._by_conf.get(conference, []))

        if len(tids_conf) < DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE:
            return None

        used: set[str] = set()
        qual: List[TeamStandingRecord] = []

        for div in divs[:2]:
            tbl = self.division_table(conference, div)

            for rec in tbl[:division_auto_qualifiers]:
                if rec.team_id not in used:
                    qual.append(rec)
                    used.add(rec.team_id)

        rest = [
            self.records[tid]
            for tid in tids_conf
            if tid in self.records and tid not in used
        ]

        rest.sort(key=self._sort_key, reverse=True)

        for rec in rest[:wildcards]:
            if rec.team_id not in used:
                qual.append(rec)
                used.add(rec.team_id)

        expected = division_auto_qualifiers * 2 + wildcards

        if len(qual) != expected:
            return None

        return qual

    def wildcard_table(self, conference: str) -> List[TeamStandingRecord]:
        """
        Return remaining teams after division top 3, sorted by wildcard race.
        """

        divs = self.divisions_for_conference(conference)
        used: set[str] = set()

        for div in divs[:2]:
            for rec in self.division_table(conference, div)[:DEFAULT_DIVISION_AUTO_QUALIFIERS]:
                used.add(rec.team_id)

        rest = [
            self.records[tid]
            for tid in self._by_conf.get(conference, [])
            if tid in self.records and tid not in used
        ]

        return sorted(rest, key=self._sort_key, reverse=True)

    def playoff_status_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a frontend-friendly status map for every team.

        Labels:
        - president
        - division_leader
        - division_seed
        - wildcard
        - bubble
        - chasing
        - longshot
        - eliminated
        """

        status: Dict[str, Dict[str, Any]] = {}

        league = self.league_table()
        ranks = self.rank_maps()

        for rec in league:
            status[rec.team_id] = {
                "playoff_status": "chasing",
                "playoff_label": "Chasing",
                "league_rank": ranks["league"].get(rec.team_id),
                "conference_rank": ranks["conference"].get(rec.team_id),
                "division_rank": ranks["division"].get(rec.team_id),
                "wildcard_rank": None,
                "is_playoff_team": False,
                "is_division_seed": False,
                "is_wildcard": False,
                "is_bubble": False,
                "is_eliminated": False,
                "clinched_note": None,
            }

        if not self._by_conf:
            top = self.league_table()[:16]
            playoff_ids = {r.team_id for r in top}

            for idx, rec in enumerate(top, start=1):
                status[rec.team_id].update(
                    {
                        "playoff_status": "league_seed",
                        "playoff_label": f"League Seed #{idx}",
                        "is_playoff_team": True,
                    }
                )

            self._mark_simple_bubble(status, league, playoff_ids, cutoff_size=16)
            return status

        for conf in self.conferences():
            eight = self.nhl_conference_playoff_eight(conf)

            if eight is None:
                seeded = self.conference_table(conf)[:DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE]
                playoff_ids = {r.team_id for r in seeded}

                for idx, rec in enumerate(seeded, start=1):
                    status[rec.team_id].update(
                        {
                            "playoff_status": "conference_seed",
                            "playoff_label": f"{conf} Seed #{idx}",
                            "is_playoff_team": True,
                        }
                    )

                conf_tbl = self.conference_table(conf)
                self._mark_simple_bubble(status, conf_tbl, playoff_ids, cutoff_size=8)
                continue

            playoff_ids = {r.team_id for r in eight}

            divs = self.divisions_for_conference(conf)

            for div in divs[:2]:
                div_tbl = self.division_table(conf, div)

                for idx, rec in enumerate(div_tbl[:3], start=1):
                    label = f"{div} #{idx}"

                    if idx == 1:
                        label = f"{div} Leader"

                    status[rec.team_id].update(
                        {
                            "playoff_status": "division_seed",
                            "playoff_label": label,
                            "is_playoff_team": True,
                            "is_division_seed": True,
                        }
                    )

            wc_tbl = self.wildcard_table(conf)

            for idx, rec in enumerate(wc_tbl, start=1):
                if rec.team_id not in status:
                    continue

                status[rec.team_id]["wildcard_rank"] = idx

                if idx <= 2:
                    status[rec.team_id].update(
                        {
                            "playoff_status": "wildcard",
                            "playoff_label": f"Wild Card {idx}",
                            "is_playoff_team": True,
                            "is_wildcard": True,
                        }
                    )
                elif idx <= 5:
                    status[rec.team_id].update(
                        {
                            "playoff_status": "bubble",
                            "playoff_label": f"Bubble: WC {idx}",
                            "is_bubble": True,
                        }
                    )
                elif rec.max_possible_points(self.season_games) < self._conference_cutline_points(conf):
                    status[rec.team_id].update(
                        {
                            "playoff_status": "eliminated",
                            "playoff_label": "Eliminated",
                            "is_eliminated": True,
                        }
                    )
                else:
                    status[rec.team_id].update(
                        {
                            "playoff_status": "longshot",
                            "playoff_label": "Longshot",
                        }
                    )

        leader = self.presidents_trophy_winner()
        if leader and leader.team_id in status:
            status[leader.team_id]["presidents_trophy_position"] = True

        self._apply_clinch_style_notes(status)
        return status

    def _mark_simple_bubble(
        self,
        status: Dict[str, Dict[str, Any]],
        table: List[TeamStandingRecord],
        playoff_ids: set,
        cutoff_size: int,
    ) -> None:
        for idx, rec in enumerate(table, start=1):
            if rec.team_id in playoff_ids:
                continue

            distance = idx - cutoff_size

            if distance <= 3:
                status[rec.team_id].update(
                    {
                        "playoff_status": "bubble",
                        "playoff_label": "Bubble",
                        "is_bubble": True,
                    }
                )
            elif rec.max_possible_points(self.season_games) < table[min(len(table) - 1, cutoff_size - 1)].points:
                status[rec.team_id].update(
                    {
                        "playoff_status": "eliminated",
                        "playoff_label": "Eliminated",
                        "is_eliminated": True,
                    }
                )
            else:
                status[rec.team_id].update(
                    {
                        "playoff_status": "longshot",
                        "playoff_label": "Longshot",
                    }
                )

    def _conference_cutline_points(self, conference: str) -> int:
        tbl = self.conference_table(conference)

        if len(tbl) < DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE:
            return 0

        return tbl[DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE - 1].points

    def _apply_clinch_style_notes(self, status: Dict[str, Dict[str, Any]]) -> None:
        """
        Simple clinch-like labels.

        This is intentionally not perfect NHL clinch math, but gives
        useful frontend flavor without requiring every remaining matchup.
        """

        for rec in self.league_table():
            row = status.get(rec.team_id)

            if not row:
                continue

            if row.get("is_eliminated"):
                row["clinched_note"] = "Mathematically outside realistic playoff range"
                continue

            if not row.get("is_playoff_team"):
                continue

            conf = rec.conference

            if not conf:
                continue

            cutline = self._conference_cutline_points(conf)

            if rec.points > cutline + 12 and rec.games_remaining(self.season_games) <= 10:
                row["clinched_note"] = "Playoff berth nearly locked"
            elif rec.points > cutline + 6 and rec.games_remaining(self.season_games) <= 6:
                row["clinched_note"] = "Strong playoff position"

            if row.get("division_rank") == 1:
                div_tbl = self.division_table(rec.conference, rec.division) if rec.division else []

                if len(div_tbl) >= 2:
                    second = div_tbl[1]
                    if rec.points > second.max_possible_points(self.season_games):
                        row["clinched_note"] = "Clinched division"

    # ------------------------------------------------------------------
    # Games back / cutline helpers
    # ------------------------------------------------------------------

    def games_back_from(self, rec: TeamStandingRecord, leader: TeamStandingRecord) -> float:
        """
        Hockey standings usually use points, but frontend sports tables
        often want a GB-like number.

        Approximation:
        Every 2 points equals 1 game back.
        """

        diff = max(0, leader.points - rec.points)
        return round(diff / 2.0, 1)

    def points_back_from(self, rec: TeamStandingRecord, target: TeamStandingRecord) -> int:
        return max(0, target.points - rec.points)

    def games_back_maps(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        league = self.league_table()
        league_leader = league[0] if league else None

        for rec in league:
            out[rec.team_id] = {
                "points_back_league": 0,
                "games_back_league": 0.0,
                "points_back_conference": 0,
                "games_back_conference": 0.0,
                "points_back_division": 0,
                "games_back_division": 0.0,
                "points_back_wildcard": None,
                "games_back_wildcard": None,
            }

            if league_leader:
                out[rec.team_id]["points_back_league"] = self.points_back_from(rec, league_leader)
                out[rec.team_id]["games_back_league"] = self.games_back_from(rec, league_leader)

        for conf in self.conferences():
            conf_tbl = self.conference_table(conf)

            if not conf_tbl:
                continue

            leader = conf_tbl[0]

            for rec in conf_tbl:
                out[rec.team_id]["points_back_conference"] = self.points_back_from(rec, leader)
                out[rec.team_id]["games_back_conference"] = self.games_back_from(rec, leader)

            cutline = conf_tbl[min(len(conf_tbl) - 1, DEFAULT_PLAYOFF_TEAMS_PER_CONFERENCE - 1)]

            for rec in conf_tbl:
                if rec.points >= cutline.points:
                    out[rec.team_id]["points_back_wildcard"] = 0
                    out[rec.team_id]["games_back_wildcard"] = 0.0
                else:
                    out[rec.team_id]["points_back_wildcard"] = self.points_back_from(rec, cutline)
                    out[rec.team_id]["games_back_wildcard"] = self.games_back_from(rec, cutline)

        for (conf, div), _tids in self._by_div.items():
            div_tbl = self.division_table(conf, div)

            if not div_tbl:
                continue

            leader = div_tbl[0]

            for rec in div_tbl:
                out[rec.team_id]["points_back_division"] = self.points_back_from(rec, leader)
                out[rec.team_id]["games_back_division"] = self.games_back_from(rec, leader)

        return out

    # ------------------------------------------------------------------
    # First-round playoff pairings
    # ------------------------------------------------------------------

    def nhl_conference_first_round_series(self, conference: str) -> List["PlayoffSeries"]:
        """
        First-round matchups for a 2-division conference:
        - Better division winner vs worse wild card
        - Other division winner vs better wild card
        - 2nd vs 3rd within each division
        """

        from .playoffs import PlayoffSeries

        eight = self.nhl_conference_playoff_eight(conference)

        if eight is None or len(eight) != 8:
            return []

        divs = self.divisions_for_conference(conference)

        if len(divs) < 2:
            return []

        div_a, div_b = divs[0], divs[1]

        div_a_tbl = self.division_table(conference, div_a)
        div_b_tbl = self.division_table(conference, div_b)

        if len(div_a_tbl) < 3 or len(div_b_tbl) < 3:
            return []

        a1, a2, a3 = div_a_tbl[:3]
        b1, b2, b3 = div_b_tbl[:3]

        used = {
            a1.team_id,
            a2.team_id,
            a3.team_id,
            b1.team_id,
            b2.team_id,
            b3.team_id,
        }

        wc = [r for r in eight if r.team_id not in used]

        if len(wc) != 2:
            return []

        wc.sort(key=self._sort_key, reverse=True)

        wc_better, wc_worse = wc[0], wc[1]

        div_winners = sorted([a1, b1], key=self._sort_key, reverse=True)

        w_hi, w_lo = div_winners[0], div_winners[1]

        def _series(
            r1: TeamStandingRecord,
            r2: TeamStandingRecord,
            hi_seed: int,
            lo_seed: int,
        ) -> PlayoffSeries:
            hi, lo = (r1, r2) if self._sort_key(r1) >= self._sort_key(r2) else (r2, r1)

            return PlayoffSeries(
                round_index=1,
                conference=conference,
                seed_high=hi_seed,
                seed_low=lo_seed,
                team_high_id=hi.team_id,
                team_low_id=lo.team_id,
            )

        out: List[PlayoffSeries] = []

        out.append(_series(w_hi, wc_worse, 1, 8))
        out.append(_series(w_lo, wc_better, 2, 7))
        out.append(_series(a2, a3, 3, 6))
        out.append(_series(b2, b3, 4, 5))

        return out

    def uses_nhl_playoff_pairings(self) -> bool:
        if not self._by_conf:
            return False

        for conf in self.conferences():
            if len(self.nhl_conference_first_round_series(conf)) != 4:
                return False

        return True

    # ------------------------------------------------------------------
    # Frontend payloads
    # ------------------------------------------------------------------

    def as_table_rows(self) -> List[Dict[str, Any]]:
        """
        Return plain JSON-friendly standings rows.

        This replaces the older simple output with:
        - league/conference/division/wildcard rank
        - record strings
        - last 10
        - streak
        - form
        - playoff labels
        - points back
        - games remaining
        - max possible points
        """

        rows: List[Dict[str, Any]] = []

        ranks = self.rank_maps()
        status = self.playoff_status_map()
        games_back = self.games_back_maps()

        for rec in self.league_table():
            base = rec.to_base_row(self.season_games)

            base.update(
                {
                    "league_rank": ranks["league"].get(rec.team_id),
                    "conference_rank": ranks["conference"].get(rec.team_id),
                    "division_rank": ranks["division"].get(rec.team_id),
                }
            )

            base.update(games_back.get(rec.team_id, {}))
            base.update(status.get(rec.team_id, {}))

            rows.append(base)

        return rows

    def as_conference_rows(self, conference: str) -> List[Dict[str, Any]]:
        ranks = self.rank_maps()
        status = self.playoff_status_map()
        games_back = self.games_back_maps()

        rows: List[Dict[str, Any]] = []

        for rec in self.conference_table(conference):
            base = rec.to_base_row(self.season_games)

            base.update(
                {
                    "league_rank": ranks["league"].get(rec.team_id),
                    "conference_rank": ranks["conference"].get(rec.team_id),
                    "division_rank": ranks["division"].get(rec.team_id),
                }
            )

            base.update(games_back.get(rec.team_id, {}))
            base.update(status.get(rec.team_id, {}))

            rows.append(base)

        return rows

    def as_division_rows(self, conference: str, division: str) -> List[Dict[str, Any]]:
        ranks = self.rank_maps()
        status = self.playoff_status_map()
        games_back = self.games_back_maps()

        rows: List[Dict[str, Any]] = []

        for rec in self.division_table(conference, division):
            base = rec.to_base_row(self.season_games)

            base.update(
                {
                    "league_rank": ranks["league"].get(rec.team_id),
                    "conference_rank": ranks["conference"].get(rec.team_id),
                    "division_rank": ranks["division"].get(rec.team_id),
                }
            )

            base.update(games_back.get(rec.team_id, {}))
            base.update(status.get(rec.team_id, {}))

            rows.append(base)

        return rows

    def playoff_picture(self) -> Dict[str, Any]:
        """
        Frontend-ready playoff picture.

        Shape:
        {
          conferences: {
            "Eastern": {
              divisions: {...},
              wildcards: [...],
              field: [...],
              bubble: [...]
            }
          }
        }
        """

        out: Dict[str, Any] = {
            "uses_nhl_pairings": self.uses_nhl_playoff_pairings(),
            "conferences": {},
        }

        status = self.playoff_status_map()

        if not self._by_conf:
            rows = self.as_table_rows()
            out["conferences"]["ALL"] = {
                "field": rows[:16],
                "bubble": rows[16:22],
                "wildcards": [],
                "divisions": {},
            }
            return out

        for conf in self.conferences():
            conf_rows = self.as_conference_rows(conf)

            div_payload: Dict[str, Any] = {}

            for div in self.divisions_for_conference(conf):
                div_payload[div] = {
                    "rows": self.as_division_rows(conf, div),
                    "top_three": self.as_division_rows(conf, div)[:3],
                }

            wc_rows = []

            for idx, rec in enumerate(self.wildcard_table(conf), start=1):
                row = rec.to_base_row(self.season_games)
                row.update(status.get(rec.team_id, {}))
                row["wildcard_rank"] = idx
                wc_rows.append(row)

            field_ids = set()

            eight = self.nhl_conference_playoff_eight(conf)

            if eight:
                field_ids = {r.team_id for r in eight}
            else:
                field_ids = {r["team_id"] for r in conf_rows[:8]}

            field = [r for r in conf_rows if r["team_id"] in field_ids]
            bubble = [
                r
                for r in conf_rows
                if not r.get("is_playoff_team") and not r.get("is_eliminated")
            ][:5]

            out["conferences"][conf] = {
                "field": field,
                "bubble": bubble,
                "wildcards": wc_rows,
                "divisions": div_payload,
                "cutline_points": self._conference_cutline_points(conf),
            }

        return out

    def league_summary(self) -> Dict[str, Any]:
        tbl = self.league_table()

        if not tbl:
            return {
                "team_count": 0,
                "leader": None,
                "last_place": None,
                "average_points": 0,
                "average_goals_for": 0,
                "average_goals_against": 0,
            }

        leader = tbl[0]
        last = tbl[-1]

        avg_points = round(sum(r.points for r in tbl) / len(tbl), 2)
        avg_gf = round(sum(r.gf for r in tbl) / len(tbl), 2)
        avg_ga = round(sum(r.ga for r in tbl) / len(tbl), 2)

        return {
            "team_count": len(tbl),
            "leader": leader.to_base_row(self.season_games),
            "last_place": last.to_base_row(self.season_games),
            "average_points": avg_points,
            "average_goals_for": avg_gf,
            "average_goals_against": avg_ga,
            "highest_scoring_team": max(tbl, key=lambda r: r.gf).to_base_row(self.season_games),
            "best_defensive_team": min(tbl, key=lambda r: r.ga).to_base_row(self.season_games),
            "best_goal_diff_team": max(tbl, key=lambda r: r.goal_diff()).to_base_row(self.season_games),
            "worst_goal_diff_team": min(tbl, key=lambda r: r.goal_diff()).to_base_row(self.season_games),
        }

    def full_payload(self) -> Dict[str, Any]:
        """
        One-stop payload for frontend screens.

        Use this if your backend wants to expose a rich standings object.
        """

        return {
            "season_games": self.season_games,
            "league_summary": self.league_summary(),
            "league_table": self.as_table_rows(),
            "playoff_picture": self.playoff_picture(),
            "conferences": {
                conf: {
                    "rows": self.as_conference_rows(conf),
                    "divisions": {
                        div: self.as_division_rows(conf, div)
                        for div in self.divisions_for_conference(conf)
                    },
                    "wildcard": [
                        rec.to_base_row(self.season_games)
                        for rec in self.wildcard_table(conf)
                    ],
                }
                for conf in self.conferences()
            },
        }