"""Apply hand-kept roster corrections after the real-NHL import.

NHL.com roster responses are cached, so a trade made after the cache was filled
(Brady Tkachuk to Florida, June 2026) would otherwise leave the player on his old
club. Each entry moves the player onto the named team's NHL roster.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

CORRECTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "real_nhl_roster_corrections.json"
_POOLS = ("roster", "ahl_roster", "echl_roster", "prospect_pool", "injured_reserve")


def _abbr(team: Any) -> str:
    return str(getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or "").upper()


def _nhl_id(p: Any) -> int:
    for attr in ("nhl_player_id", "external_player_id"):
        try:
            v = int(str(getattr(p, attr, "") or "0"))
            if v:
                return v
        except (TypeError, ValueError):
            continue
    pid = str(getattr(p, "id", "") or "")
    if pid.startswith("NHL_") and pid[4:].isdigit():
        return int(pid[4:])
    return 0


def load_corrections() -> Dict[int, Dict[str, Any]]:
    try:
        raw = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {int(k): v for k, v in raw.items() if str(k).isdigit() and isinstance(v, dict) and v.get("team")}


def apply_roster_corrections(teams: List[Any]) -> List[str]:
    fixes = load_corrections()
    if not fixes:
        return []
    by_abbr = {_abbr(t): t for t in teams or []}
    moved: List[str] = []
    for team in teams or []:
        for pool in _POOLS:
            lst = list(getattr(team, pool, None) or [])
            for p in lst:
                fix = fixes.get(_nhl_id(p))
                if not fix:
                    continue
                dest = by_abbr.get(str(fix["team"]).upper())
                if dest is None or dest is team:
                    continue
                try:
                    getattr(team, pool).remove(p)
                    dest_roster = getattr(dest, "roster", None)
                    if dest_roster is None:
                        dest.roster = []
                        dest_roster = dest.roster
                    dest_roster.append(p)
                    from services.real_nhl_roster_importer import _assign_player_to_nhl

                    _assign_player_to_nhl(p, dest)  # same bookkeeping the importer uses (minors flags, team context)
                    moved.append(f"{fix.get('name') or getattr(p, 'id', '')}: {_abbr(team)} -> {_abbr(dest)}")
                except Exception:
                    log.debug("roster correction failed", exc_info=True)
    return moved


def _ovr(p: Any) -> float:
    fn = getattr(p, "ovr", None)
    try:
        return float(fn()) if callable(fn) else float(fn or 0)
    except Exception:
        return 0.0


def keep_corrected_on_nhl_roster(teams: List[Any]) -> int:
    """After the 23-man trim, keep corrected players in the NHL lineup.

    The trim keeps the highest-rated 23, which would drop a corrected player with a
    low house-rule rating straight to the AHL. Swap him with the weakest NHL skater.
    """
    fixes = load_corrections()
    if not fixes:
        return 0
    swapped = 0
    for team in teams or []:
        ahl = list(getattr(team, "ahl_roster", None) or [])
        roster = list(getattr(team, "roster", None) or [])
        for p in ahl:
            fix = fixes.get(_nhl_id(p))
            if not fix or str(fix.get("team", "")).upper() != _abbr(team) or fix.get("level", "nhl") != "nhl":
                continue
            skaters = [r for r in roster if str(getattr(getattr(r, "position", None), "value", getattr(r, "position", "")) or "").upper() != "G"]
            if not skaters:
                continue
            weakest = min(skaters, key=_ovr)
            try:
                team.ahl_roster.remove(p)
                team.roster.remove(weakest)
                team.roster.append(p)
                team.ahl_roster.append(weakest)
                from services.real_nhl_roster_importer import _assign_player_to_ahl, _assign_player_to_nhl

                _assign_player_to_nhl(p, team)
                _assign_player_to_ahl(weakest, team)
                roster = list(team.roster)
                swapped += 1
            except Exception:
                log.debug("pin to NHL roster failed", exc_info=True)
    return swapped
