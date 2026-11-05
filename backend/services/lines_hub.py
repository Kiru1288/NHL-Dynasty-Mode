"""Lines screen hub payload: the real numbers behind the Lines overhaul.

One call returns what the board shows next to each player and unit:
season stats, on-ice results for every forward trio / D pair you have used
("points together"), the next opponent and his top units, special-teams
percentages and goalie usage against the coach's plan.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


def unit_book_for_season(session: Any, create: bool = True) -> Optional[Dict[str, Any]]:
    """Even-strength unit results for the current season, keyed "F|id|id|id" / "D|id|id"."""
    season = str(int(getattr(session, "season_calendar_year", 0) or 0))
    book = getattr(session, "_unit_book", None)
    if not isinstance(book, dict):
        if not create:
            return None
        book = {}
        session._unit_book = book
    cur = book.get(season)
    if cur is None and create:
        cur = {}
        book[season] = cur
        # keep two seasons
        for old in sorted(book.keys())[:-2]:
            book.pop(old, None)
    return cur


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _unit_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    xgf, xga = _f(rec.get("xgf")), _f(rec.get("xga"))
    sf, sa = int(rec.get("sf") or 0), int(rec.get("sa") or 0)
    return {
        "gp": int(rec.get("gp") or 0),
        "gf": int(rec.get("gf") or 0),
        "ga": int(rec.get("ga") or 0),
        "pts": int(rec.get("pts") or 0),
        "sf": sf,
        "sa": sa,
        "xgf_pct": round(100.0 * xgf / (xgf + xga), 1) if (xgf + xga) > 0.5 else None,
        "sf_pct": round(100.0 * sf / (sf + sa), 1) if (sf + sa) >= 5 else None,
    }


def _player_lines(session: Any, team: Any) -> Dict[str, Dict[str, Any]]:
    reg = getattr(session, "player_season_stats", None) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for p in list(getattr(team, "roster", None) or []) + list(getattr(team, "injured_reserve", None) or []):
        pid = str(getattr(p, "id", "") or "")
        row = reg.get(pid) if pid else None
        if not isinstance(row, dict):
            continue
        gp = int(row.get("gp") or 0)
        pos = str(row.get("position") or "").upper()
        line: Dict[str, Any] = {"gp": gp}
        if pos == "G":
            sa = int(row.get("shots_against") or 0)
            saves = int(row.get("saves") or max(0, sa - int(row.get("ga") or 0)))
            toi = int(row.get("toi_sec") or 0)
            line.update({
                "w": int(row.get("w") or 0),
                "l": int(row.get("l") or 0),
                "otl": int(row.get("otl") or 0),
                "sv_pct": round(saves / sa, 3) if sa > 0 else None,
                "gaa": round(int(row.get("ga") or 0) * 3600.0 / toi, 2) if toi > 0 else None,
                "so": int(row.get("so") or row.get("shutouts") or 0),
            })
        else:
            toi = int(row.get("toi_sec") or 0)
            line.update({
                "g": int(row.get("g") or 0),
                "a": int(row.get("a") or 0),
                "pts": int(row.get("pts") or 0),
                "pm": int(row.get("plus_minus") or 0),
                "ppp": int(row.get("ppg") or 0) + int(row.get("ppa") or 0),
                "sog": int(row.get("sog") or 0),
                "toi_pg": round(toi / gp / 60.0, 1) if gp > 0 and toi > 0 else None,
            })
        out[pid] = line
    return out


def _next_game(session: Any, F: Any) -> Optional[Dict[str, Any]]:
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    uid = str(getattr(session, "user_team_id", "") or "")
    by_day = getattr(session, "by_day", None) or {}
    cal = getattr(session, "nhl_calendar", None) or []
    for idx in range(cur, min(len(cal), cur + 40)):
        for sl in list(by_day.get(idx, []) or []):
            hid = F._safe_slot_team_id(sl, "home_id")
            aid = F._safe_slot_team_id(sl, "away_id")
            if uid not in (hid, aid):
                continue
            opp_id = aid if hid == uid else hid
            opp = session.team_by_id.get(str(opp_id))
            out: Dict[str, Any] = {
                "date": F._calendar_iso_for_day(session, idx),
                "days_away": idx - cur,
                "home": hid == uid,
                "opp_id": str(opp_id),
                "opp_name": str(getattr(opp, "name", "") or ""),
                "opp_abbr": str(F._franchise_team_abbrev(opp) if opp is not None else ""),
            }
            if opp is not None:
                try:
                    lines = F._synthetic_even_strength_lines_for_team(opp, getattr(session, "player_season_stats", None))
                    by_id = {F._canonical_lineup_player_id(p): p for p in getattr(opp, "roster", None) or []}

                    def unit(slots: Dict[str, Any]) -> Dict[str, Any]:
                        ps = [by_id.get(str(v)) for v in (slots or {}).values() if v]
                        ps = [p for p in ps if p is not None]
                        ovr = [int(round(F._player_ovr99(p))) for p in ps]
                        return {
                            "names": [F._name_str(p).split(" ")[-1] for p in ps],
                            "ovr": round(sum(ovr) / len(ovr)) if ovr else None,
                        }

                    fwd = lines.get("forwards") or []
                    dee = lines.get("defense") or []
                    out["top_line"] = unit((fwd[0] or {}).get("slots") or {}) if fwd else None
                    out["top_pair"] = unit((dee[0] or {}).get("slots") or {}) if dee else None
                except Exception:
                    _log.debug("opponent units failed", exc_info=True)
                try:
                    rows = F._build_standings_rows(session)
                    row = next((r for r in rows if str(r.get("team_id")) == str(opp_id)), None)
                    if row:
                        out["opp_record"] = f"{int(row.get('w') or 0)}-{int(row.get('l') or 0)}-{int(row.get('otl') or 0)}"
                        out["opp_gf"] = row.get("gf")
                        out["opp_ga"] = row.get("ga")
                except Exception:
                    pass
            return out
    return None


def build_lines_hub(session: Any) -> Dict[str, Any]:
    from services import franchise_sim as F

    uid = str(getattr(session, "user_team_id", "") or "")
    team = session.team_by_id.get(uid)
    if team is None:
        return {"ok": False}
    out: Dict[str, Any] = {"ok": True, "season": int(getattr(session, "season_calendar_year", 0) or 0)}
    out["players"] = _player_lines(session, team)
    book = unit_book_for_season(session, create=False) or {}
    out["units"] = {k: _unit_row(v) for k, v in book.items() if isinstance(v, dict) and int(v.get("gp") or 0) > 0}
    try:
        out["next_game"] = _next_game(session, F)
    except Exception:
        _log.debug("next game failed", exc_info=True)
        out["next_game"] = None
    try:
        rows = F._build_standings_rows(session)
        mine = next((r for r in rows if str(r.get("team_id")) == uid), {}) or {}
        out["team"] = {
            "record": f"{int(mine.get('w') or 0)}-{int(mine.get('l') or 0)}-{int(mine.get('otl') or 0)}",
            "gf": mine.get("gf"),
            "ga": mine.get("ga"),
            "pp_pct": mine.get("pp_pct"),
            "pk_pct": mine.get("pk_pct") if mine.get("pk_pct") is not None else mine.get("penalty_kill_pct"),
        }
    except Exception:
        _log.debug("standings failed", exc_info=True)
        out["team"] = {}
    try:
        eng = getattr(session, "sim", None)
        hist = eng._gm_goalie_start_state(team) if eng is not None else {}
        lines_root = getattr(session, "lines", None) or {}
        es = lines_root.get("even_strength") if isinstance(lines_root, dict) else None
        out["goalies"] = {
            "plan": (es or {}).get("goalie_plan") if isinstance(es, dict) else None,
            "starts": {
                str(k): {
                    "gp": int((v or {}).get("gp") or 0),
                    "consecutive": int((v or {}).get("consecutive") or 0),
                    "recent_sv": (v or {}).get("recent_save_pct"),
                }
                for k, v in (hist or {}).items()
                if not str(k).startswith("_") and isinstance(v, dict)
            },
        }
    except Exception:
        _log.debug("goalie usage failed", exc_info=True)
        out["goalies"] = {}
    return out
