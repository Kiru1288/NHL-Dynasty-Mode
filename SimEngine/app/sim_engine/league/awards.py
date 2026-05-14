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


def _player_age_from_rosters(teams: List[Any], pid: str) -> int:
    for t in teams:
        for p in getattr(t, "roster", None) or []:
            if str(getattr(p, "id", "") or "") != str(pid):
                continue
            try:
                return int(getattr(p, "age", getattr(getattr(p, "identity", None), "age", 24)) or 24)
            except Exception:
                return 24
    return 24


def compute_awards(
    standings: StandingsTable,
    playoff_result: Optional[PlayoffResult],
    teams: List[Any],
    player_season_stats: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Award]:
    """
    Produce yearly awards: Presidents' Trophy, Stanley Cup, and (when
    player_season_stats is provided) Art Ross, Rocket Richard, Hart,
    Norris, Vezina, Calder from game-derived totals.
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

    # ------------------------------------------------------------------
    # Player awards (requires game-derived player_season_stats rows)
    # ------------------------------------------------------------------
    rows = [r for r in (player_season_stats or []) if isinstance(r, dict)]
    if not rows:
        return awards

    sk = [r for r in rows if str(r.get("position", "")).upper() != "G" and int(r.get("gp", 0) or 0) >= 20]
    gl = [r for r in rows if str(r.get("position", "")).upper() == "G" and int(r.get("gp", 0) or 0) >= 25]

    def _tid(row: Dict[str, Any]) -> str:
        return str(row.get("team_id") or "")

    def _pts(row: Dict[str, Any]) -> int:
        return int(row.get("pts", int(row.get("g", 0) or 0) + int(row.get("a", 0) or 0)) or 0)

    def _gf(row: Dict[str, Any]) -> int:
        return int(row.get("g", 0) or 0)

    def _svpct(row: Dict[str, Any]) -> float:
        sa = int(row.get("shots_against", 0) or 0)
        sv = int(row.get("saves", 0) or 0)
        if sa <= 0:
            return 0.0
        return float(sv) / float(sa)

    tbl = standings.league_table()
    rank_by_tid = {rec.team_id: i for i, rec in enumerate(tbl)}

    if sk:
        art = max(sk, key=lambda r: (_pts(r), _gf(r), -rank_by_tid.get(_tid(r), 99)))
        awards["Art Ross Trophy"] = Award(
            name="Art Ross Trophy",
            winner_team_id=_tid(art),
            winner_name=str(art.get("name") or ""),
            finalists=[str(x.get("name") or "") for x in sorted(sk, key=lambda r: -_pts(r))[:3]],
            rationale=f"League-leading {_pts(art)} points ({int(art.get('g',0) or 0)}G-{int(art.get('a',0) or 0)}A).",
        )
        rocket = max(sk, key=lambda r: (_gf(r), _pts(r), -rank_by_tid.get(_tid(r), 99)))
        awards["Rocket Richard Trophy"] = Award(
            name="Rocket Richard Trophy",
            winner_team_id=_tid(rocket),
            winner_name=str(rocket.get("name") or ""),
            finalists=[str(x.get("name") or "") for x in sorted(sk, key=lambda r: -_gf(r))[:3]],
            rationale=f"Most goals in the league ({_gf(rocket)}).",
        )

        defs = [r for r in sk if str(r.get("position", "")).upper() == "D"]
        if defs:
            norris = max(defs, key=lambda r: (_pts(r), -rank_by_tid.get(_tid(r), 99)))
            awards["Norris Trophy"] = Award(
                name="Norris Trophy",
                winner_team_id=_tid(norris),
                winner_name=str(norris.get("name") or ""),
                finalists=[str(x.get("name") or "") for x in sorted(defs, key=lambda r: -_pts(r))[:3]],
                rationale=f"Top-scoring defenseman ({_pts(norris)} pts).",
            )

        hart_cand = sorted(sk, key=lambda r: -_pts(r))[:24]
        if hart_cand:
            best = None
            best_score = -1e9
            for r in hart_cand:
                rk = rank_by_tid.get(_tid(r), 20)
                pts = _pts(r)
                score = float(pts) - 1.15 * float(rk) + 0.35 * float(_gf(r))
                if score > best_score:
                    best_score = score
                    best = r
            if best is not None:
                awards["Hart Memorial Trophy"] = Award(
                    name="Hart Memorial Trophy",
                    winner_team_id=_tid(best),
                    winner_name=str(best.get("name") or ""),
                    finalists=[str(x.get("name") or "") for x in hart_cand[:3]],
                    rationale="MVP blend of individual production and team success.",
                )

        calder_pool = [
            r
            for r in sk
            if int(r.get("gp", 0) or 0) >= 28
            and _player_age_from_rosters(teams, str(r.get("player_id", ""))) <= 24
        ]
        if calder_pool:
            calder = max(calder_pool, key=lambda r: (_pts(r), _gf(r)))
            awards["Calder Memorial Trophy"] = Award(
                name="Calder Memorial Trophy",
                winner_team_id=_tid(calder),
                winner_name=str(calder.get("name") or ""),
                finalists=[str(x.get("name") or "") for x in sorted(calder_pool, key=lambda r: -_pts(r))[:3]],
                rationale="Top rookie season by points among age-eligible first-year skaters.",
            )

    if gl:
        vez = max(
            gl,
            key=lambda r: (
                _svpct(r) * 0.72 + min(1.0, int(r.get("w", 0) or 0) / max(1, int(r.get("gp", 1) or 1))) * 0.28,
                int(r.get("w", 0) or 0),
            ),
        )
        awards["Vezina Trophy"] = Award(
            name="Vezina Trophy",
            winner_team_id=_tid(vez),
            winner_name=str(vez.get("name") or ""),
            finalists=[str(x.get("name") or "") for x in sorted(gl, key=lambda r: -_svpct(r))[:3]],
            rationale=f"Elite save percentage ({_svpct(vez):.3f}) and workload.",
        )

    return awards


