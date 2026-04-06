from __future__ import annotations

"""
League awards logic.

This module is intentionally conservative: it awards only what the current
data can justify, and degrades gracefully when detailed player/coach stats
are not yet available.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .standings import StandingsTable, TeamStandingRecord
from .playoffs import PlayoffResult


@dataclass
class Award:
    name: str
    winner_team_id: Optional[str] = None
    winner_name: Optional[str] = None
    finalists: List[str] = None
    rationale: str = ""


def _team_name_from_id(teams: Dict[str, Any], tid: str, default: Optional[str] = None) -> str:
    t = teams.get(tid)
    if t is None:
        return default or tid
    name = getattr(t, "name", None)
    city = getattr(t, "city", None)
    if city and name:
        return f"{city} {name}"
    if name:
        return str(name)
    return default or tid


def compute_awards(
    standings: StandingsTable,
    playoff_result: Optional[PlayoffResult],
    teams: List[Any],
) -> Dict[str, Award]:
    """
    Produce a minimal but structured set of yearly awards.

    Always tries to award:
        - Presidents' Trophy (best regular-season team)
        - Stanley Cup Champion (from playoffs)

    If more detailed stats become available in the future, this function
    can be extended to include Art Ross / Rocket / Hart / Norris / Vezina
    using Player_Stats, but for now this module stays team-centric.
    """
    awards: Dict[str, Award] = {}
    team_map = {}
    for t in teams:
        tid = getattr(t, "team_id", None) or getattr(t, "id", None)
        if tid is not None:
            team_map[str(tid)] = t

    # Presidents' Trophy -------------------------------------------------
    prez = standings.presidents_trophy_winner()
    if prez is not None:
        name = _team_name_from_id(team_map, prez.team_id, prez.name)
        rationale = f"Best regular-season record ({prez.points} pts, {prez.wins}-{prez.losses}-{prez.otl}, GD {prez.goal_diff():+d})."
        awards["Presidents Trophy"] = Award(
        name="Presidents Trophy",
        winner_team_id=prez.team_id,
        winner_name=name,
        finalists=[],
        rationale=rationale,
    )

    # Stanley Cup Champion -----------------------------------------------
    if playoff_result is not None:
        champ_id = playoff_result.champion_id
        champ_name = _team_name_from_id(team_map, champ_id)
        rationale = "Won the Stanley Cup after navigating the playoff bracket."
        awards["Stanley Cup"] = Award(
            name="Stanley Cup",
            winner_team_id=champ_id,
            winner_name=champ_name,
            finalists=[_team_name_from_id(team_map, tid) for tid in playoff_result.finalist_ids if tid != champ_id],
            rationale=rationale,
        )

    # Conference titles (if applicable) ----------------------------------
    if playoff_result is not None:
        # Heuristically derive conference champions from finalists if
        # conference metadata exists. For now, we just mirror finalists.
        for tid in playoff_result.finalist_ids:
            tname = _team_name_from_id(team_map, tid)
            if "Conference" not in tname:
                continue
        # This area intentionally left conservative until conferences
        # are richer in upstream models.

    return awards


