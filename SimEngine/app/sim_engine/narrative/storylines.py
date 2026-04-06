"""
Storylines — league-wide narrative arcs (teams, dynasties, rivalries).

Tracks dynasty detection, contender windows, rebuild success, collapse, rivalry.
Observes standings and playoff results only; does not alter outcomes.
Deterministic given the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Storyline types
STORYLINE_DYNASTY = "dynasty"
STORYLINE_CONTENDER_WINDOW = "contender_window"
STORYLINE_REBUILD = "rebuild"
STORYLINE_COLLAPSE = "collapse"
STORYLINE_RIVALRY = "rivalry"
STORYLINE_HISTORIC_SEASON = "historic_season"
STORYLINE_REDEMPTION_ARC = "redemption_arc"


@dataclass
class TeamStorylineState:
    """Per-team state for storyline tracking."""
    team_id: str
    team_name: str
    cup_years: List[int] = field(default_factory=list)
    recent_standings_rank: List[int] = field(default_factory=list)
    recent_points: List[int] = field(default_factory=list)
    storyline_tags: List[str] = field(default_factory=list)


def _standings_rank(standings: List[Any], team_id: str) -> Optional[int]:
    for i, s in enumerate(standings):
        tid = getattr(s, "team_id", None) or getattr(s, "id", None)
        if str(tid) == str(team_id):
            return i + 1
    return None


def _team_name_from_standing(s: Any) -> str:
    return getattr(s, "team_name", None) or getattr(s, "name", None) or str(getattr(s, "team_id", ""))


def update_storylines(
    standings: List[Any],
    playoff_champion_id: Optional[str],
    state: Any,
    year: int,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Update storyline state from standings and playoff result; emit narrative events.
    context["recent_cup_winners"] = [(team_id, year), ...]
    context["team_storylines"] = dict[team_id -> TeamStorylineState]
    Returns list of {"type": "storyline", "teams", "message", "storyline_type"}.
    """
    events: List[Dict[str, Any]] = []
    recent_cups: List[Tuple[str, int]] = context.get("recent_cup_winners") or []
    team_states: Dict[str, TeamStorylineState] = context.get("team_storylines") or {}
    context["team_storylines"] = team_states
    context["recent_cup_winners"] = recent_cups

    if playoff_champion_id:
        recent_cups.append((str(playoff_champion_id), year))
    recent_cups[:] = [(tid, y) for tid, y in recent_cups if year - y <= 10]

    n_teams = len(standings) or 1
    for i, s in enumerate(standings or []):
        tid = str(getattr(s, "team_id", None) or getattr(s, "id", "") or "")
        tname = _team_name_from_standing(s)
        pts = int(getattr(s, "points", 0) or 0)
        rank = i + 1

        ts = team_states.get(tid)
        if ts is None:
            ts = TeamStorylineState(team_id=tid, team_name=tname)
            team_states[tid] = ts

        ts.recent_standings_rank.append(rank)
        ts.recent_points.append(pts)
        if len(ts.recent_standings_rank) > 8:
            ts.recent_standings_rank.pop(0)
            ts.recent_points.pop(0)

        cup_count_5y = sum(1 for cid, y in recent_cups if cid == tid and year - y <= 5)
        cup_count_6y = sum(1 for cid, y in recent_cups if cid == tid and year - y <= 6)

        # --- Dynasty: 3+ cups in 5–6 years
        if (cup_count_5y >= 3 or cup_count_6y >= 3) and STORYLINE_DYNASTY not in ts.storyline_tags:
            ts.storyline_tags.append(STORYLINE_DYNASTY)
            events.append({
                "type": "storyline",
                "teams": [tname],
                "message": f"The {tname} are building a dynasty after capturing their third Stanley Cup in five seasons.",
                "storyline_type": STORYLINE_DYNASTY,
            })
        elif cup_count_5y >= 2 and STORYLINE_DYNASTY not in ts.storyline_tags:
            ts.storyline_tags.append(STORYLINE_DYNASTY)
            events.append({
                "type": "storyline",
                "teams": [tname],
                "message": f"The {tname} appear to be entering a dynasty era after winning another championship.",
                "storyline_type": STORYLINE_DYNASTY,
            })

        # --- Contender window: top 8 finish 3+ years
        if len(ts.recent_standings_rank) >= 3 and all(r <= 8 for r in ts.recent_standings_rank[-3:]):
            if STORYLINE_CONTENDER_WINDOW not in ts.storyline_tags:
                ts.storyline_tags.append(STORYLINE_CONTENDER_WINDOW)
                events.append({
                    "type": "storyline",
                    "teams": [tname],
                    "message": f"The {tname} have emerged as one of the league's perennial contenders.",
                    "storyline_type": STORYLINE_CONTENDER_WINDOW,
                })

        # --- Rebuild success: was bottom 5, now top 12 (playoff range)
        if len(ts.recent_standings_rank) >= 4:
            old_ranks = ts.recent_standings_rank[:-1]
            was_bottom = any(r >= n_teams - 4 for r in old_ranks[-3:])
            now_top = rank <= min(12, n_teams // 2 + 2)
            if was_bottom and now_top and rank <= 16 and STORYLINE_REBUILD not in ts.storyline_tags:
                ts.storyline_tags.append(STORYLINE_REBUILD)
                events.append({
                    "type": "storyline",
                    "teams": [tname],
                    "message": f"After years of rebuilding, the {tname} have finally returned to playoff contention.",
                    "storyline_type": STORYLINE_REBUILD,
                })

        # --- Collapse: was top 5, now bottom 10
        if len(ts.recent_standings_rank) >= 4:
            old_ranks = ts.recent_standings_rank[:-1]
            was_top = any(r <= 5 for r in old_ranks[-3:])
            now_bottom = rank >= max(10, n_teams - 10)
            if was_top and now_bottom and STORYLINE_COLLAPSE not in ts.storyline_tags:
                ts.storyline_tags.append(STORYLINE_COLLAPSE)
                events.append({
                    "type": "storyline",
                    "teams": [tname],
                    "message": f"The once-dominant {tname} appear to be entering a period of decline.",
                    "storyline_type": STORYLINE_COLLAPSE,
                })

        # --- Redemption arc: bottom-feeder climb after a tagged collapse
        if (
            STORYLINE_COLLAPSE in ts.storyline_tags
            and STORYLINE_REDEMPTION_ARC not in ts.storyline_tags
            and len(ts.recent_standings_rank) >= 3
        ):
            worst_recent = max(ts.recent_standings_rank[-3:])
            if worst_recent >= max(8, n_teams - 6) and rank <= min(10, max(6, n_teams // 3)):
                ts.storyline_tags.append(STORYLINE_REDEMPTION_ARC)
                events.append({
                    "type": "storyline",
                    "teams": [tname],
                    "message": (
                        f"The {tname} are climbing out of the basement - a redemption story written in standings points."
                    ),
                    "storyline_type": STORYLINE_REDEMPTION_ARC,
                })

    # --- Rivalry: two teams repeatedly near top (top 4 or close in standings)
    if len(standings) >= 4:
        top4 = [str(getattr(s, "team_id", None) or getattr(s, "id", "")) for s in standings[:4]]
        for tid in top4:
            ts = team_states.get(tid)
            if not ts or len(ts.recent_standings_rank) < 3:
                continue
            if all(r <= 6 for r in ts.recent_standings_rank[-3:]) and STORYLINE_RIVALRY not in ts.storyline_tags:
                others = [team_states[t].team_name for t in top4 if t != tid and t in team_states]
                if others:
                    ts.storyline_tags.append(STORYLINE_RIVALRY)
                    events.append({
                        "type": "storyline",
                        "teams": [ts.team_name, others[0]],
                        "message": f"A fierce rivalry is developing between the {ts.team_name} and {others[0]} after multiple playoff clashes.",
                        "storyline_type": STORYLINE_RIVALRY,
                    })
                break

    return events
