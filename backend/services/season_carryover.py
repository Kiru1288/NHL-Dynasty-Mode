"""Season-to-season continuity.

At rollover the finished season is archived instead of thrown away:
* every player's regular-season line is appended to ``player.career_stats["seasons"]``
  (same shape as the real NHL import), so dossiers, Calder eligibility, contracts and
  trade value see a continuous career;
* a compact copy of the season ledger is kept on ``session.player_season_archive``;
* each club's record plus analytics (shots, xG, Corsi, sh%, sv%, PDO, goals vs xG) goes
  into the season history row, and ``session.team_luck_carryover`` holds the regression
  read (a lucky club is projected to give points back, an unlucky one to gain).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

# Every simulated season is kept (no cap). Rows are compacted to the stat keys that
# matter so a long save stays small: see ``_compact_archive_row``.
ARCHIVE_SEASONS = None
_ARCHIVE_DROP_KEYS = ("team_abbrev", "source", "league", "season", "ties")


def _compact_archive_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Archive row without duplicated / derivable keys, None values or zero counters."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k in _ARCHIVE_DROP_KEYS or v is None or v == "":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0 and k not in ("gp",):
            continue
        out[k] = v
    return out


def season_label(year: int) -> str:
    return f"{int(year)}-{(int(year) + 1) % 100:02d}"


def _i(v: Any) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0


def _f(v: Any) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _players_by_id(session: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from services.franchise_sim import _iter_league_players_for_aging

        for p in _iter_league_players_for_aging(getattr(session.sim, "league", None)):
            pid = str(getattr(p, "id", "") or getattr(p, "player_id", "") or "")
            if pid:
                out[pid] = p
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return out


def _abbr(session: Any, tid: str) -> str:
    t = (getattr(session, "team_by_id", None) or {}).get(str(tid))
    return str(getattr(t, "abbreviation", None) or getattr(t, "abbr", None) or tid)


def _career_line(session: Any, row: Dict[str, Any], *, label: str, team_id: str, counts: Dict[str, Any], team_label: str) -> Dict[str, Any]:
    pos = str(row.get("position") or "").upper()
    goalie = pos.startswith("G")
    gp = _i(counts.get("gp", row.get("gp")))
    line: Dict[str, Any] = {
        "season": label,
        "team": team_label,
        "team_abbrev": team_label,
        "league": "NHL",
        "gp": gp,
        "source": "sim",
        "team_id": str(team_id or ""),
    }
    if goalie:
        sa = _i(counts.get("shots_against", row.get("shots_against") or row.get("goalie_shots_against")))
        ga = _i(counts.get("ga", row.get("ga") or row.get("goalie_ga")))
        toi = _i(counts.get("toi_sec", row.get("toi_sec")))
        line.update({
            "wins": _i(counts.get("w", row.get("w"))),
            "losses": _i(counts.get("l", row.get("l"))),
            "otl": _i(counts.get("otl", row.get("otl"))),
            "ties": _i(counts.get("t", row.get("t"))),
            "gs": _i(counts.get("gs", row.get("gs") or row.get("starts"))),
            "sv_pct": round(1 - ga / sa, 3) if sa else None,
            "gaa": round(ga * 3600 / toi, 2) if toi else None,
            "shutouts": _i(counts.get("so", row.get("so"))),
        })
    else:
        g, a = _i(counts.get("g", row.get("g"))), _i(counts.get("a", row.get("a")))
        line.update({
            "g": g, "a": a, "pts": g + a,
            "plus_minus": _i(counts.get("plus_minus", row.get("plus_minus") or row.get("pm"))),
            "pim": _i(counts.get("pim", row.get("pim"))),
            "sog": _i(counts.get("sog", row.get("sog"))),
            "toi_per_gp": round(_i(counts.get("toi_sec", row.get("toi_sec"))) / 60.0 / max(1, gp), 1),
        })
    return line


def archive_player_lines(session: Any, season_year: int) -> int:
    from services.franchise_sim import _row_team_slices

    pss = getattr(session, "player_season_stats", None) or {}
    label = season_label(season_year)
    players = _players_by_id(session)
    n = 0
    compact: Dict[str, Dict[str, Any]] = {}
    for pid, row in pss.items():
        if not isinstance(row, dict) or _i(row.get("gp")) <= 0:
            continue
        pos = str(row.get("position") or "").upper()
        slices = [(tid, counts) for tid, counts in _row_team_slices(row) if tid and _i(counts.get("gp")) > 0]
        if not slices:
            slices = [(str(row.get("team_id") or ""), {"gp": _i(row.get("gp"))})]
        lines_out: List[Dict[str, Any]] = []
        for tid, counts in slices:
            lines_out.append(_career_line(session, row, label=label, team_id=tid, counts=counts, team_label=_abbr(session, tid)))
        if len(lines_out) > 1:
            lines_out.append(_career_line(
                session, row, label=label, team_id=str(row.get("team_id") or ""),
                counts=row, team_label="TOT",
            ))
        line = lines_out[-1] if len(lines_out) > 1 else lines_out[0]
        compact[str(pid)] = _compact_archive_row({**line, "name": row.get("name"), "position": pos, "team_id": str(row.get("team_id") or "")})
        p = players.get(str(pid))
        if p is None:
            continue
        cs = getattr(p, "career_stats", None)
        cs = dict(cs) if isinstance(cs, dict) else {}
        seasons: List[Dict[str, Any]] = [s for s in list(cs.get("seasons") or []) if isinstance(s, dict)]
        if any(str(s.get("season")) == label and s.get("source") == "sim" for s in seasons):
            pass
        else:
            seasons = [s for s in seasons if str(s.get("season")) != label]
            seasons.extend(lines_out)
            cs["seasons"] = seasons
        po = (getattr(session, "playoff_player_stats", None) or {}).get(str(pid))
        if isinstance(po, dict) and _i(po.get("gp")) > 0:
            po_line = _career_line(
                session, po, label=label, team_id=str(po.get("team_id") or row.get("team_id") or ""),
                counts=po, team_label=_abbr(session, str(po.get("team_id") or row.get("team_id") or "")),
            )
            po_line["league"] = "NHL"
            po_seasons = [s for s in list(cs.get("playoff_seasons") or []) if isinstance(s, dict)]
            if not any(str(s.get("season")) == label and s.get("source") == "sim" for s in po_seasons):
                po_seasons = [s for s in po_seasons if str(s.get("season")) != label]
                po_seasons.append(po_line)
                cs["playoff_seasons"] = po_seasons
        try:
            setattr(p, "career_stats", cs)
            n += 1
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    try:
        n += archive_minor_league_lines(session, season_year, players)
    except Exception:
        _swallowed_log.warning("minor-league career archive failed", exc_info=True)
    arch = dict(getattr(session, "player_season_archive", None) or {})
    arch[str(season_year)] = compact
    # No season cap: older blocks from saves made before compaction are slimmed in place.
    for k, block in list(arch.items()):
        if isinstance(block, dict) and any(isinstance(r, dict) and "team_abbrev" in r for r in block.values()):
            arch[k] = {pid: _compact_archive_row(r) for pid, r in block.items() if isinstance(r, dict)}
    session.player_season_archive = arch
    return n


def _append_career_line(p: Any, line: Dict[str, Any]) -> bool:
    cs = getattr(p, "career_stats", None)
    cs = dict(cs) if isinstance(cs, dict) else {}
    seasons = [x for x in list(cs.get("seasons") or []) if isinstance(x, dict)]
    key = (str(line.get("season")), str(line.get("league") or "").upper())
    if any((str(x.get("season")), str(x.get("league") or "NHL").upper()) == key for x in seasons):
        return False
    seasons.append(line)
    cs["seasons"] = seasons
    try:
        setattr(p, "career_stats", cs)
    except Exception:
        return False
    return True


def archive_minor_league_lines(session: Any, season_year: int, players: Optional[Dict[str, Any]] = None) -> int:
    """AHL and junior/NCAA/Europe lines go into career_stats too, so the dossier keeps
    every level a player skated at (they used to vanish at rollover)."""
    label = season_label(season_year)
    players = players if players is not None else _players_by_id(session)
    try:
        from services.ahl_league import get_player_ahl_season_line
    except Exception:
        get_player_ahl_season_line = None  # type: ignore
    try:
        from services.franchise_sim import _draft_league_display
    except Exception:
        _draft_league_display = lambda c, n="": str(c or n or "Junior")  # type: ignore
    n = 0
    for pid, p in players.items():
        pos = str(getattr(getattr(getattr(p, "identity", None), "position", None), "value", "") or getattr(getattr(p, "identity", None), "position", "") or "").upper()
        goalie = pos.startswith("G")
        if get_player_ahl_season_line is not None:
            try:
                al = get_player_ahl_season_line(session, pid, season_year)
            except Exception:
                al = None
            if isinstance(al, dict) and _i(al.get("gp")) > 0:
                team = str(al.get("team") or al.get("team_name") or "AHL")
                line: Dict[str, Any] = {"season": label, "team": team, "team_abbrev": team, "league": "AHL", "gp": _i(al.get("gp")), "source": "ahl_sim"}
                if goalie:
                    line.update({"wins": _i(al.get("wins")), "losses": _i(al.get("losses")), "otl": _i(al.get("ot_losses")),
                                 "sv_pct": al.get("save_pct"), "gaa": al.get("gaa"), "shutouts": _i(al.get("shutouts"))})
                else:
                    g, a = _i(al.get("goals")), _i(al.get("assists"))
                    line.update({"g": g, "a": a, "pts": g + a, "plus_minus": _i(al.get("plus_minus")), "pim": _i(al.get("pim"))})
                if _append_career_line(p, line):
                    n += 1
        if getattr(p, "real_nhl_import", False) or getattr(p, "real_player", False):
            continue
        st = getattr(p, "_prospect_season_stats", None)
        if not isinstance(st, dict) or _i(st.get("gp") or st.get("games_played")) <= 0:
            continue
        sy_line = _i(getattr(p, "_prospect_season_year", 0))
        if sy_line and sy_line != int(season_year):
            continue
        asg = getattr(p, "_franchise_assignment", None) or {}
        code = str((asg.get("league_code") if isinstance(asg, dict) else "") or st.get("league") or getattr(p, "current_league_id", "") or "")
        if not code or code.upper() in ("NHL", "AHL", "ECHL"):
            continue
        club = str((asg.get("club") if isinstance(asg, dict) else "") or st.get("team") or "")
        line = {"season": label, "team": club or code, "team_abbrev": club or code, "league": _draft_league_display(code), "league_code": code,
                "gp": _i(st.get("gp") or st.get("games_played")), "source": "junior_sim"}
        if goalie:
            line.update({"wins": _i(st.get("wins")), "sv_pct": st.get("sv_pct") or st.get("save_pct"), "gaa": st.get("gaa")})
        else:
            g, a = _i(st.get("goals")), _i(st.get("assists"))
            line.update({"g": g, "a": a, "pts": _i(st.get("points")) or g + a, "pim": _i(st.get("pim"))})
        if _append_career_line(p, line):
            n += 1
    return n


def restore_archived_nhl_lines(session: Any, p: Any, pid: str) -> int:
    """Re-add NHL season lines from ``session.player_season_archive`` that an older build
    stripped from career_stats (the dossier gap: 2025-26 and 2027-28 but no 2026-27)."""
    arch = getattr(session, "player_season_archive", None) or {}
    if not isinstance(arch, dict) or not arch:
        return 0
    keys = {str(pid), str(getattr(p, "id", "") or "")}
    if str(pid).startswith("NHL_"):
        keys.add(str(pid)[4:])
    added = 0
    for sy in sorted(arch.keys()):
        block = arch.get(sy) or {}
        row = next((block.get(k) for k in keys if isinstance(block.get(k), dict)), None)
        if not row or _i(row.get("gp")) <= 0:
            continue
        line = {k: v for k, v in row.items() if k not in ("name", "position")}
        try:
            line.setdefault("season", season_label(int(sy)))
        except (TypeError, ValueError):
            continue
        line.setdefault("team_abbrev", line.get("team") or "")
        line.setdefault("league", "NHL")
        if "g" in line or "a" in line or "pts" in line:
            line.setdefault("pts", _i(line.get("g")) + _i(line.get("a")))
        line["source"] = "sim"
        if _append_career_line(p, line):
            added += 1
    if added:
        cs = dict(getattr(p, "career_stats", None) or {})
        cs["seasons"] = sorted(cs.get("seasons") or [], key=lambda x: str(x.get("season") or ""))
        setattr(p, "career_stats", cs)
    return added


def team_season_analytics(session: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for g in list(getattr(session, "game_results", None) or []):
        if str(g.get("stat_scope") or "regular_season") != "regular_season":
            continue
        for side, opp in (("home", "away"), ("away", "home")):
            tid = str(g.get(f"{side}_id") or "")
            if not tid:
                continue
            r = out.setdefault(tid, {"gp": 0, "gf": 0, "ga": 0, "sf": 0, "sa": 0, "xgf": 0.0, "xga": 0.0, "cf": 0, "ca": 0, "ppg": 0, "ppo": 0, "ppga": 0, "pk_opp": 0})
            r["gp"] += 1
            r["gf"] += _i(
                g.get(f"player_{side}_goals", g.get(f"hockey_{side}_goals", g.get(f"{side}_score", g.get(f"{side}_goals"))))
            )
            r["ga"] += _i(
                g.get(f"player_{opp}_goals", g.get(f"hockey_{opp}_goals", g.get(f"{opp}_score", g.get(f"{opp}_goals"))))
            )
            r["sf"] += _i(g.get(f"{side}_sog"))
            r["sa"] += _i(g.get(f"{opp}_sog"))
            r["xgf"] += _f(g.get(f"{side}_xg"))
            r["xga"] += _f(g.get(f"{opp}_xg"))
            r["cf"] += _i(g.get(f"{side}_cf"))
            r["ca"] += _i(g.get(f"{opp}_cf"))
            r["ppg"] += _i(g.get(f"{side}_pp_goals"))
            r["ppo"] += _i(g.get(f"{side}_ppo"))
            r["ppga"] += _i(g.get(f"{side}_ppga"))
            r["pk_opp"] += _i(g.get(f"{side}_opp_ppo"))
    for tid, r in out.items():
        sh = r["gf"] / r["sf"] if r["sf"] else 0.0
        sv = 1 - r["ga"] / r["sa"] if r["sa"] else 0.0
        r.update({
            "xgf": round(r["xgf"], 1), "xga": round(r["xga"], 1),
            "sh_pct": round(sh * 100, 1), "sv_pct": round(sv, 3), "pdo": round((sh + sv) * 1000, 1),
            "cf_pct": round(r["cf"] / (r["cf"] + r["ca"]) * 100, 1) if r["cf"] + r["ca"] else None,
            "xgf_pct": round(r["xgf"] / (r["xgf"] + r["xga"]) * 100, 1) if r["xgf"] + r["xga"] else None,
            "gf_minus_xgf": round(r["gf"] - r["xgf"], 1), "xga_minus_ga": round(r["xga"] - r["ga"], 1),
            "pp_pct": round(r["ppg"] / r["ppo"] * 100, 1) if r["ppo"] else None,
            "pk_pct": round((1 - r["ppga"] / r["pk_opp"]) * 100, 1) if r["pk_opp"] else None,
        })
        # goals above expectation at both ends → points of "luck" (≈ 6 goals per standings win, 2 pts)
        luck_goals = r["gf_minus_xgf"] + r["xga_minus_ga"]
        r["luck_goals"] = round(luck_goals, 1)
        r["regression_pts"] = round(-luck_goals / 6.0 * 2.0 * 0.5, 1)  # half of it is expected to fade
        r["luck_label"] = "Lucky" if luck_goals >= 12 else "Unlucky" if luck_goals <= -12 else "Neutral"
    return out


def archive_completed_season(session: Any, history_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Call once at rollover, before the season ledger/standings/game log are cleared."""
    sy = int(getattr(session, "season_calendar_year", 0) or 0)
    if int(getattr(session, "_season_archived_year", 0) or 0) == sy:
        return history_entry
    archive_player_lines(session, sy)
    analytics = team_season_analytics(session)
    teams: Dict[str, Dict[str, Any]] = {}
    recs = getattr(getattr(session, "standings", None), "records", None) or {}
    for tid, rr in (recs.items() if isinstance(recs, dict) else []):
        tid = str(tid)
        w, l, o = _i(getattr(rr, "wins", 0)), _i(getattr(rr, "losses", 0)), _i(getattr(rr, "otl", 0))
        teams[tid] = {"abbr": _abbr(session, tid), "w": w, "l": l, "otl": o, "pts": _i(getattr(rr, "points", 2 * w + o)),
                      "gf": _i(getattr(rr, "gf", 0)), "ga": _i(getattr(rr, "ga", 0)), **{k: v for k, v in (analytics.get(tid) or {}).items() if k not in ("gp", "gf", "ga")}}
    order = sorted(teams, key=lambda t: (-teams[t]["pts"], t))
    for i, t in enumerate(order):
        teams[t]["league_rank"] = i + 1
    history_entry["season_label"] = season_label(sy)
    history_entry["teams"] = teams
    try:
        aw = getattr(session, "awards_payload", None) or {}
        winners = []
        src = aw.get("official_results") or aw.get("awards") or aw.get("trophies") or []
        for a in list(src.values() if isinstance(src, dict) else src):
            if isinstance(a, dict):
                winners.append({k: a.get(k) for k in ("award_id", "id", "name", "winner_name", "player_name", "winner_id", "player_id", "team_id", "team_abbr")})
        history_entry["awards"] = winners
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    session.team_luck_carryover = {t: {"season": sy, "pdo": r.get("pdo"), "luck_goals": r.get("luck_goals"), "regression_pts": r.get("regression_pts"),
                                        "luck_label": r.get("luck_label"), "xgf_pct": r.get("xgf_pct"), "pts": r.get("pts")} for t, r in teams.items()}
    try:
        import os

        if os.environ.get("NHL_LEGACY_DISABLED") != "1":  # perf A/B and emergency off switch
            from services.legacy_hall import archive_legacy_season

            archive_legacy_season(session, sy)
    except Exception:
        _swallowed_log.warning("legacy season archive failed", exc_info=True)
    session._season_archived_year = sy
    return history_entry


def _franchise_first_season(session: Any, season_year: int) -> int:
    """Earliest season this franchise save has simulated (sticky on the session)."""
    cands: List[int] = [int(season_year)]
    stored = getattr(session, "_franchise_first_season_year", None)
    if isinstance(stored, int) and stored > 0:
        cands.append(stored)
    for k in (getattr(session, "player_season_archive", None) or {}).keys():
        if _i(k) > 0:
            cands.append(_i(k))
    for h in list(getattr(session, "season_history", None) or []):
        if isinstance(h, dict) and _i(h.get("season_year")) > 0:
            cands.append(_i(h.get("season_year")))
    first = min(cands)
    try:
        session._franchise_first_season_year = first
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return first


def _age_on_sept15(p: Any, season_year: int) -> Optional[int]:
    ident = getattr(p, "identity", None)
    by = _i(getattr(ident, "birth_year", 0) or getattr(p, "birth_year", 0))
    bm = _i(getattr(ident, "birth_month", 0))
    bd = _i(getattr(ident, "birth_day", 0))
    bdate = str(getattr(p, "birth_date", "") or "")
    if bdate[:4].isdigit():
        parts = [x for x in bdate.split("-")[:3] if x.isdigit()]
        if len(parts) == 3:
            by, bm, bd = int(parts[0]), int(parts[1]), int(parts[2])
    if by <= 0:
        return None
    age = int(season_year) - by
    if bm and (bm, bd or 1) > (9, 15):
        age -= 1
    return age


def calder_history(session: Any, season_year: int, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Player award history merged with NHL rookie facts for Calder eligibility.

    NHL rule: a player is NOT a rookie if, in any single preceding season, he played
    more than 25 NHL games, or if he played 6+ NHL games in each of any two preceding
    seasons. He must also be under 26 on Sept. 15 of the season (``age_sept15``).

    Prior NHL seasons come from ``player.career_stats["seasons"]`` (real-NHL import
    lines plus the lines archived at every franchise rollover). Generated players have
    no pre-franchise lines, so seasons before the save began are estimated from the
    generator's entry pattern (drafted at 18, NHL debut at draft year + 3, ~age 21);
    players who were in a prospect pool (junior/AHL) when the save began get none.
    """
    out = dict(base or {})
    cur = season_label(season_year)
    first = _franchise_first_season(session, season_year)
    for pid, p in _players_by_id(session).items():
        cs = getattr(p, "career_stats", None)
        seasons = list(cs.get("seasons") or []) if isinstance(cs, dict) else []
        by_season: Dict[str, int] = {}
        for s in seasons:
            if not isinstance(s, dict):
                continue
            if str(s.get("league") or "NHL").upper() != "NHL" or str(s.get("season") or "") >= cur:
                continue
            by_season[str(s.get("season"))] = by_season.get(str(s.get("season")), 0) + _i(s.get("gp"))
        basis = "recorded NHL seasons"
        estimated = 0
        real_import = bool(getattr(p, "real_nhl_import", False))
        if not by_season and real_import:
            imp = getattr(p, "real_nhl_import_stats", None) or {}
            prior_gp = 0
            if isinstance(imp, dict):
                prior_gp = _i(imp.get("gp") or imp.get("gamesPlayed") or imp.get("games"))
                # The importer falls back to the CURRENT season's line when the player
                # has no last-season NHL games (pick_stats_row secondary). Those are
                # this season's games, not a prior season: a year-1 rookie (Celebrini
                # in a 2024-25 save) was being disqualified by his own rookie year.
                imp_season = _i(str(imp.get("seasonId") or imp.get("season_id") or imp.get("season") or "")[:4])
                if imp_season and imp_season >= int(season_year):
                    prior_gp = 0
            if prior_gp > 0:
                by_season["import-rating-season"] = prior_gp
                basis = "imported NHL games (rating-year stats)"
            else:
                basis = "real NHL import; career seasons not loaded yet"
        if not by_season and not real_import:
            pre_known = any(k < season_label(first) for k in by_season)
            in_prospect_pool = getattr(p, "_prospect_season_year", None) is not None
            if not pre_known and not in_prospect_pool:
                ident = getattr(p, "identity", None)
                dy = _i(getattr(ident, "draft_year", 0))
                a0 = _age_on_sept15(p, first)
                # Age 23 and under with no recorded NHL seasons are still rookies.
                # Inventing a full 82-game career emptied the Calder ballot.
                if a0 is not None and a0 <= 23:
                    estimated = 0
                elif dy > 0:
                    estimated = max(0, first - dy - 3)
                else:
                    estimated = max(0, (a0 or 0) - 23)
                if estimated > 0:
                    by_season["pre-franchise-0"] = 30
                    basis = f"estimated {estimated} NHL season(s) before the save began"
        elif not by_season and real_import:
            pre_known = any(k < season_label(first) for k in by_season)
            in_prospect_pool = getattr(p, "_prospect_season_year", None) is not None
            if not pre_known and not in_prospect_pool:
                ident = getattr(p, "identity", None)
                dy = _i(getattr(ident, "draft_year", 0))
                a0 = _age_on_sept15(p, first)
                if a0 is not None and a0 <= 23:
                    estimated = 0
                elif dy > 0:
                    estimated = max(0, first - dy - 3)
                else:
                    estimated = max(0, (a0 or 0) - 23)
                if estimated > 0:
                    by_season["pre-franchise-0"] = 30
                    basis = f"estimated {estimated} NHL season(s) before the save began"
        max_gp = max(by_season.values(), default=0)
        six_plus = sum(1 for v in by_season.values() if v >= 6)
        rookie = max_gp <= 25 and six_plus < 2
        if rookie:
            why = f"Rookie: max {max_gp} GP in a prior season ({basis})."
        elif max_gp > 25:
            why = f"Not a rookie: played {max_gp} NHL games in a prior season ({basis})."
        else:
            why = f"Not a rookie: 6+ NHL games in {six_plus} prior seasons ({basis})."
        row = dict(out.get(pid) or {}) if isinstance(out.get(pid), dict) else {}
        row.update(
            {
                "is_rookie": rookie,
                "prior_nhl_gp": sum(by_season.values()),
                "prior_nhl_seasons": len(by_season),
                "max_prior_season_gp": max_gp,
                "prior_seasons_6plus_gp": six_plus,
                "estimated_pre_franchise_seasons": estimated,
                "calder_basis": why,
            }
        )
        age = _age_on_sept15(p, season_year)
        if age is not None:
            row["age_sept15"] = age
        out[pid] = row
    return out


def last_season_payload(session: Any) -> Dict[str, Any]:
    hist = [h for h in list(getattr(session, "season_history", None) or []) if isinstance(h, dict) and h.get("teams")]
    if not hist:
        return {"available": False}
    h = hist[-1]
    return {"available": True, "season_year": h.get("season_year"), "season_label": h.get("season_label"), "teams": h.get("teams"),
            "awards": h.get("awards") or [], "luck": getattr(session, "team_luck_carryover", None) or {}}
