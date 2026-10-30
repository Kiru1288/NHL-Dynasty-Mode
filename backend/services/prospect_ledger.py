"""Prospect stats ledger for Stats Central: generated prospects only, grouped by league."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

_log = logging.getLogger(__name__)


def _age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    try:
        return int(getattr(ident, "age", None) or getattr(p, "age", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _pos(p: Any) -> str:
    ident = getattr(p, "identity", None)
    raw = getattr(ident, "position", None) or getattr(p, "position", "")
    return str(getattr(raw, "value", raw) or "").upper()


def _name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", "") or "?")


def build_prospect_ledger(session: Any) -> Dict[str, Any]:
    league = getattr(getattr(session, "sim", None), "league", None)
    day = int(getattr(session, "calendar_cursor", 0) or 0)
    # Rights move on draft day and in trades without the calendar moving; key on those too.
    ds = getattr(session, "draft_state", None) or {}
    ckey = (
        day,
        str(getattr(session, "phase", "") or ""),
        len(ds.get("completed_picks") or []) if isinstance(ds, dict) else 0,
        int(getattr(session, "_stats_revision", 0) or 0),
        len(list(getattr(session, "trade_history", None) or [])),
    )
    cache = getattr(session, "_prospect_ledger_cache", None)
    if isinstance(cache, dict) and cache.get("day") == ckey and cache.get("payload"):
        return cache["payload"]
    try:
        from app.sim_engine.generation.prospect_league_scoring import ensure_prospect_season_stats
    except Exception:  # pragma: no cover
        ensure_prospect_season_stats = None
    cal_iso = None
    try:
        from services.franchise_sim import _calendar_iso_for_day

        cal_iso = _calendar_iso_for_day(session, day)
    except Exception:
        _log.debug("calendar iso failed", exc_info=True)
    season_year = int(getattr(session, "season_calendar_year", 0) or 0) or None
    draft_year = (season_year or 0) + 1
    rng = getattr(getattr(session, "sim", None), "rng", None)
    teams = getattr(session, "team_by_id", None) or {}
    try:
        from app.sim_engine.league_hierarchy_bootstrap import _player_sept15_age as _sept15_age
    except Exception:
        _sept15_age = None

    rows: List[Dict[str, Any]] = []
    real_dev: List[Any] = []
    leagues: Dict[str, str] = {}
    for block in list(getattr(league, "development_leagues", None) or []):
        code = str(block.get("league_code") or "")
        lname = str(block.get("league_name") or code)
        leagues[code] = lname
        for tm in block.get("teams") or []:
            club = str(tm.get("name") or tm.get("team_name") or "")
            for p in tm.get("players") or []:
                if getattr(p, "retired", False):
                    continue
                if getattr(p, "real_nhl_import", False):
                    real_dev.append((p, code, lname, club))
                    continue
                _pre = getattr(p, "_prospect_season_stats", None)
                pre_line = dict(_pre) if isinstance(_pre, dict) else None
                st = None
                if ensure_prospect_season_stats is not None:
                    try:
                        st = ensure_prospect_season_stats(p, code, rng=rng, calendar_iso=cal_iso, season_year=season_year)
                    except Exception:
                        st = None
                cached = pre_line
                if not isinstance(st, dict) or (
                    isinstance(cached, dict)
                    and int(cached.get("gp") or 0) > int((st or {}).get("gp") or 0)
                ):
                    # The date-based refresh can come back empty (offseason, calendar edge);
                    # never show fewer games than the line already on file.
                    st = cached if isinstance(cached, dict) else {}
                gp = int(st.get("gp") or st.get("games_played") or 0)
                age = _age(p)
                pos = _pos(p)
                rights = str(getattr(p, "nhl_rights_team_id", None) or getattr(p, "rights_team_id", None) or "")
                # Only rights earned in a draft this franchise played count. Startup filler
                # handed generated juniors to random NHL clubs with no draft record.
                has_record = any(
                    getattr(p, k, None) not in (None, "", 0)
                    for k in ("draft_round", "draft_overall", "drafted_overall", "draft_pick", "draft_year_selected")
                ) or isinstance(getattr(p, "draft_info", None), dict) and bool((getattr(p, "draft_info") or {}).get("round"))
                if getattr(p, "_bootstrap_fake_rights", False):
                    rights = ""
                drafted = bool(rights)
                rt = teams.get(rights)
                if rights and rt is None:
                    rights = ""
                    drafted = False
                g = int(st.get("goals") or 0)
                a = int(st.get("assists") or 0)
                row = {
                    "player_id": str(getattr(p, "id", "") or ""),
                    "name": _name(p),
                    "age": age,
                    "position": pos,
                    "league_code": code,
                    "league": lname,
                    "club": club,
                    "gp": gp,
                    "g": g,
                    "a": a,
                    "pts": int(st.get("points") or (g + a)),
                    "ppg": round((g + a) / gp, 2) if gp else 0.0,
                    "pim": int(st.get("pim") or 0),
                    "drafted": drafted,
                    "rights_team": str(getattr(rt, "abbr", None) or getattr(rt, "abbreviation", None) or getattr(rt, "name", None) or "") if rt else "",
                    "draft_eligible_year": int(getattr(p, "draft_eligible_year", 0) or 0) or None,
                    "status": "",
                    "rights_team_id": rights,
                    "rights_team_abbr": str(getattr(rt, "abbr", None) or getattr(rt, "abbreviation", None) or "") if rt else "",
                    "exceptional": bool(getattr(p, "exceptional_status", False)),
                }
                elig = None
                if _sept15_age is not None:
                    for y in range(draft_year, draft_year + 5):
                        try:
                            if int(_sept15_age(p, y)) >= 18:
                                elig = y
                                break
                        except Exception:
                            break
                row["draft_year"] = elig
                if drafted:
                    row["status"] = "Drafted"
                elif row["exceptional"]:
                    row["status"] = "Exceptional"
                elif elig == draft_year:
                    first_year = True
                    if _sept15_age is not None:
                        try:
                            first_year = int(_sept15_age(p, draft_year)) <= 18
                        except Exception:
                            first_year = True
                    row["status"] = "Draft eligible" if first_year else "Overage"
                elif elig is None or elig < draft_year:
                    row["status"] = "Undrafted"
                else:
                    row["status"] = "Future class"
                if pos == "G":
                    row.update({
                        "w": int(st.get("wins") or 0),
                        "sv_pct": st.get("sv_pct") or st.get("save_pct"),
                        "gaa": st.get("gaa"),
                    })
                rows.append(row)
    # Real-world prospects held by NHL clubs: listed with their rights team, no simulated line.
    def _abbr(t: Any) -> str:
        return str(getattr(t, "abbr", None) or getattr(t, "abbreviation", None) or getattr(t, "name", "") or "")

    seen_real: set = set()

    def _add_real(p: Any, team: Any, code: str, lname: str, club: str) -> None:
        pid = str(getattr(p, "id", "") or "")
        if not pid or pid in seen_real or getattr(p, "retired", False):
            return
        seen_real.add(pid)
        asg = getattr(p, "_franchise_assignment", None) or {}
        code = code or str(asg.get("league_code") or "")
        rows.append({
            "player_id": pid, "name": _name(p), "age": _age(p), "position": _pos(p),
            "league_code": code, "league": lname or code or "—",
            "club": club or str(asg.get("club") or ""), "gp": None, "g": None, "a": None, "pts": None, "ppg": None,
            "drafted": team is not None,
            "rights_team": _abbr(team) if team is not None else "",
            "rights_team_id": str(getattr(team, "team_id", None) or getattr(team, "id", "") or "") if team is not None else "",
            "rights_team_abbr": _abbr(team) if team is not None else "",
            "status": "Drafted" if team is not None else "Undrafted", "real": True,
            "draft_year": getattr(p, "draft_year", None),
        })

    for t in list(getattr(league, "teams", None) or []):
        for p in list(getattr(t, "prospect_pool", None) or []):
            if getattr(p, "real_nhl_import", False):
                _add_real(p, t, "", "", "")
    for p, code, lname, club in real_dev:
        rid = str(getattr(p, "nhl_rights_team_id", None) or getattr(p, "rights_team_id", None) or "")
        _add_real(p, teams.get(rid), code, lname, club)
    rows.sort(key=lambda r: (-(r["pts"] or -1), -(r["g"] or 0), r["gp"] or 0))
    payload = {
        "rows": rows,
        "leagues": [{"code": k, "name": v} for k, v in sorted(leagues.items(), key=lambda kv: kv[1])],
        "draft_year": draft_year,
        "count": len(rows),
    }
    try:
        session._prospect_ledger_cache = {"day": ckey, "payload": payload}
    except Exception:
        pass
    return payload
