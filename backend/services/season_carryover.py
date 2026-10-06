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

ARCHIVE_SEASONS = 6


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
        pass
    return out


def _abbr(session: Any, tid: str) -> str:
    t = (getattr(session, "team_by_id", None) or {}).get(str(tid))
    return str(getattr(t, "abbreviation", None) or getattr(t, "abbr", None) or tid)


def archive_player_lines(session: Any, season_year: int) -> int:
    pss = getattr(session, "player_season_stats", None) or {}
    label = season_label(season_year)
    players = _players_by_id(session)
    n = 0
    compact: Dict[str, Dict[str, Any]] = {}
    for pid, row in pss.items():
        if not isinstance(row, dict) or _i(row.get("gp")) <= 0:
            continue
        pos = str(row.get("position") or "").upper()
        goalie = pos.startswith("G")
        line: Dict[str, Any] = {"season": label, "team": _abbr(session, row.get("team_id")), "team_abbrev": _abbr(session, row.get("team_id")),
                                "league": "NHL", "gp": _i(row.get("gp")), "source": "sim"}
        if goalie:
            sa = _i(row.get("shots_against") or row.get("goalie_shots_against"))
            ga = _i(row.get("ga") or row.get("goalie_ga"))
            toi = _i(row.get("toi_sec"))
            line.update({"wins": _i(row.get("w")), "losses": _i(row.get("l")), "otl": _i(row.get("otl")),
                         "sv_pct": round(1 - ga / sa, 3) if sa else None, "gaa": round(ga * 3600 / toi, 2) if toi else None,
                         "shutouts": _i(row.get("so"))})
        else:
            g, a = _i(row.get("g")), _i(row.get("a"))
            line.update({"g": g, "a": a, "pts": _i(row.get("pts")) or g + a, "plus_minus": _i(row.get("plus_minus") or row.get("pm")),
                         "pim": _i(row.get("pim")), "sog": _i(row.get("sog")), "toi_per_gp": round(_i(row.get("toi_sec")) / 60.0 / max(1, _i(row.get("gp"))), 1)})
        compact[str(pid)] = {**line, "name": row.get("name"), "position": pos, "team_id": str(row.get("team_id") or "")}
        p = players.get(str(pid))
        if p is None:
            continue
        cs = getattr(p, "career_stats", None)
        cs = dict(cs) if isinstance(cs, dict) else {}
        seasons: List[Dict[str, Any]] = [s for s in list(cs.get("seasons") or []) if isinstance(s, dict)]
        if any(str(s.get("season")) == label and s.get("source") == "sim" for s in seasons):
            continue
        # a real-import row for the same season label (shouldn't exist for sim seasons) is replaced
        seasons = [s for s in seasons if str(s.get("season")) != label]
        seasons.append(line)
        cs["seasons"] = seasons
        try:
            setattr(p, "career_stats", cs)
            n += 1
        except Exception:
            pass
    arch = dict(getattr(session, "player_season_archive", None) or {})
    arch[str(season_year)] = compact
    for k in sorted(arch.keys())[:-ARCHIVE_SEASONS]:
        arch.pop(k, None)
    session.player_season_archive = arch
    return n


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
            r["gf"] += _i(g.get(f"{side}_score", g.get(f"{side}_goals")))
            r["ga"] += _i(g.get(f"{opp}_score", g.get(f"{opp}_goals")))
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
        pass
    session.team_luck_carryover = {t: {"season": sy, "pdo": r.get("pdo"), "luck_goals": r.get("luck_goals"), "regression_pts": r.get("regression_pts"),
                                        "luck_label": r.get("luck_label"), "xgf_pct": r.get("xgf_pct"), "pts": r.get("pts")} for t, r in teams.items()}
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
        pass
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
        if not bool(getattr(p, "real_nhl_import", False)):
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
