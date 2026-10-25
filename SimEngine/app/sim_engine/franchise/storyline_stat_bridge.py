"""Bridge franchise storyline / universe modifiers into live engine.py stat allocation.

Also feeds the game stats ledger back into morale and storyline beats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _player_id(player: Any) -> str:
    return str(getattr(player, "id", "") or "")


def _roster_modifiers(team: Any) -> Dict[str, Dict[str, float]]:
    from app.sim_engine.franchise.storyline_conduct import get_player_stat_allocation_modifiers  # noqa: WPS433

    out: Dict[str, Dict[str, float]] = {}
    for player in getattr(team, "roster", None) or []:
        pid = _player_id(player)
        if not pid:
            continue
        mods = get_player_stat_allocation_modifiers(player)
        if mods:
            out[pid] = mods
    return out


def player_stat_map_from_box(box: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten home/away skater + goalie rows from a completed stats-ledger box."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(box, dict):
        return out
    for key in ("home_skaters", "away_skaters"):
        for row in list(box.get(key) or []):
            if not isinstance(row, dict):
                continue
            pid = str(row.get("player_id") or row.get("id") or "")
            if not pid:
                continue
            g = int(row.get("g") or row.get("goals") or 0)
            a = int(row.get("a") or row.get("assists") or 0)
            out[pid] = {
                "goals": g,
                "assists": a,
                "points": g + a,
                "sog": int(row.get("sog") or 0),
                "pim": int(row.get("pim") or 0),
                "hits": int(row.get("hit") or row.get("hits") or 0),
                "blocks": int(row.get("blk") or row.get("blocks") or 0),
                "toi_sec": int(row.get("toi_sec") or 0),
                "xgf": float(row.get("xgf") or 0),
                "xga": float(row.get("xga") or 0),
                "position": str(row.get("position") or ""),
                "is_goalie": False,
            }
    for key in ("home_goalie", "away_goalie"):
        row = box.get(key)
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or row.get("id") or "")
        if not pid:
            continue
        ga = int(row.get("ga") or row.get("goals_against") or 0)
        saves = int(row.get("saves") or row.get("sv") or 0)
        sa = int(row.get("sa") or row.get("shots_against") or 0) or (saves + ga)
        out[pid] = {
            "goals": 0,
            "assists": int(row.get("a") or 0),
            "points": int(row.get("a") or 0),
            "ga": ga,
            "saves": saves,
            "sa": sa,
            "shutout": bool(row.get("so") or row.get("shutout") or (ga == 0 and sa >= 1)),
            "is_goalie": True,
            "position": "G",
        }
    for key in ("home_goalies", "away_goalies"):
        for row in list(box.get(key) or []):
            if not isinstance(row, dict):
                continue
            pid = str(row.get("player_id") or row.get("id") or "")
            if not pid or pid in out:
                continue
            ga = int(row.get("ga") or 0)
            saves = int(row.get("saves") or 0)
            sa = int(row.get("sa") or 0) or (saves + ga)
            out[pid] = {
                "goals": 0,
                "assists": int(row.get("a") or 0),
                "points": int(row.get("a") or 0),
                "ga": ga,
                "saves": saves,
                "sa": sa,
                "shutout": bool(row.get("so") or (ga == 0 and sa >= 1)),
                "is_goalie": True,
                "position": "G",
            }
    return out


def apply_stats_ledger_to_storylines(session: Any, box: Dict[str, Any], rng: Any = None) -> Dict[str, Any]:
    """After a game box is written to the ledger, drive morale + published story beats."""
    if not isinstance(box, dict):
        return {"ingested": 0, "postgame": []}
    stats_map = player_stat_map_from_box(box)
    ingested = 0
    try:
        from app.sim_engine.franchise.storyline_coverage import ingest_game_box_storylines  # noqa: WPS433

        ingested = int(ingest_game_box_storylines(session, box) or 0)
    except Exception:
        ingested = 0
    hid = str(box.get("home_id") or "")
    aid = str(box.get("away_id") or "")
    hg = int(box.get("home_goals") or box.get("home_score") or 0)
    ag = int(box.get("away_goals") or box.get("away_score") or 0)
    postgame: list = []
    for tid, won in ((hid, hg > ag), (aid, ag > hg)):
        if not tid:
            continue
        payload = {
            "won": won,
            "result": "W" if won else "L",
            "game_id": box.get("game_id") or box.get("id"),
            "player_stats": stats_map,
            "home_id": hid,
            "away_id": aid,
            "home_goals": hg,
            "away_goals": ag,
            "home_skaters": box.get("home_skaters"),
            "away_skaters": box.get("away_skaters"),
            "home_goalie": box.get("home_goalie"),
            "away_goalie": box.get("away_goalie"),
        }
        try:
            from app.sim_engine.franchise.storyline_engine import apply_universe_postgame  # noqa: WPS433

            postgame.append(apply_universe_postgame(session, tid, payload, rng=rng))
        except Exception:
            continue
    return {"ingested": ingested, "postgame": postgame}


def build_franchise_game_stat_context(
    session: Any,
    home_team_id: str,
    away_team_id: str,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build per-player stat allocation modifiers for one matchup.

    Prefers narrative-universe V3 when available; otherwise maps active
    storyline conduct readiness modifiers onto stat fingerprints.
    """
    meta = dict(game_meta or {})
    try:
        from app.sim_engine.franchise.storyline_engine import build_universe_matchup_context  # noqa: WPS433

        return build_universe_matchup_context(session, home_team_id, away_team_id, meta)
    except (ImportError, AttributeError):
        pass

    team_by_id = getattr(session, "team_by_id", None) or {}
    home = team_by_id.get(str(home_team_id))
    away = team_by_id.get(str(away_team_id))
    home_mods = _roster_modifiers(home) if home is not None else {}
    away_mods = _roster_modifiers(away) if away is not None else {}
    base_id = str(meta.get("game_id") or f"conduct_{home_team_id}_{away_team_id}_{meta.get('calendar_day', 0)}")
    return {
        "id": base_id,
        "home_team_id": str(home_team_id),
        "away_team_id": str(away_team_id),
        "home": {"player_modifiers": home_mods},
        "away": {"player_modifiers": away_mods},
        "home_win_probability_delta": 0.0,
    }


def prime_franchise_game_stat_modifiers(
    session: Any,
    sim: Any,
    home_team_id: str,
    away_team_id: str,
    *,
    game_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach storyline/universe modifiers to the live SimEngine for one game."""
    context = build_franchise_game_stat_context(session, home_team_id, away_team_id, game_meta)
    home_mods = dict((context.get("home") or {}).get("player_modifiers") or {})
    away_mods = dict((context.get("away") or {}).get("player_modifiers") or {})
    fn = getattr(sim, "set_franchise_game_stat_modifiers", None)
    if callable(fn):
        fn(
            home_player_modifiers=home_mods,
            away_player_modifiers=away_mods,
            home_win_probability_delta=float(context.get("home_win_probability_delta", 0) or 0),
        )
    return context


def clear_franchise_game_stat_modifiers(sim: Any) -> None:
    fn = getattr(sim, "clear_franchise_game_stat_modifiers", None)
    if callable(fn):
        fn()


def apply_franchise_postgame_storyline(
    session: Any,
    team_id: str,
    game_result: Dict[str, Any],
    rng: Any = None,
) -> None:
    """Optional universe postgame tick when V3 helpers exist."""
    try:
        from app.sim_engine.franchise.storyline_engine import apply_universe_postgame  # noqa: WPS433
    except (ImportError, AttributeError):
        return
    try:
        apply_universe_postgame(session, team_id, game_result, rng=rng)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Box-score records (incremental max-compare per stored box; never rescans)
# ---------------------------------------------------------------------------

# Line indexes into a compact box ``lines`` row (team_id first) — see
# serialization.BOX_LINE_KEYS.
_REC_G, _REC_A, _REC_SOG = 2, 3, 6
_REC_SAVES, _REC_SA = 19, 20

# Smallest value worth a published "record" beat (all-time books start empty).
_REC_PUBLISH_MIN = {"goals": 3, "assists": 4, "points": 5, "saves": 42, "team_goals": 7, "team_shots": 48, "margin": 6}

_REC_LABEL = {
    "goals": "goals", "assists": "assists", "points": "points", "saves": "saves",
    "team_goals": "goals in a game", "team_shots": "shots in a game", "margin": "goal margin",
}


def _rec_row(value: float, compact: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    row = {
        "value": value,
        "game_id": str(compact.get("id") or ""),
        "season": int(compact.get("season") or 0),
        "date": str(compact.get("iso") or ""),
    }
    row.update(extra)
    return row


def _rec_try(book: Dict[str, Any], key: str, value: float, row_fn: Any) -> bool:
    """Max-compare one record slot. Returns True when a new record is set."""
    cur = book.get(key)
    if value <= 0 or (isinstance(cur, dict) and float(cur.get("value") or 0) >= value):
        return False
    book[key] = row_fn()
    return True


def update_box_score_records(session: Any, compact: Dict[str, Any], *, publish: bool = True) -> List[Dict[str, Any]]:
    """
    Fold one stored box into the season / franchise-history record books and
    publish a storyline when a meaningful record falls.

    Books (``session.box_score_records``):
      season_game / franchise_game   — user-team single-game player records
      league_season_game             — any player, this season
      team_season / team_alltime     — user-team single-game team records
      vs_user_season / vs_user       — most by a CPU player AGAINST the user club
    """
    if not isinstance(compact, dict) or str(compact.get("scope") or "") == "preseason":
        return []
    recs = getattr(session, "box_score_records", None)
    if not isinstance(recs, dict):
        recs = {}
        session.box_score_records = recs
    season = int(compact.get("season") or 0)
    if int(recs.get("season") or 0) != season:
        for k in ("season_game", "league_season_game", "team_season", "vs_user_season"):
            recs[k] = {}
        recs["season"] = season
    for k in ("season_game", "franchise_game", "league_season_game", "team_season", "team_alltime", "vs_user_season", "vs_user"):
        recs.setdefault(k, {})
    utid = str(getattr(session, "user_team_id", "") or "")
    hid, aid = str(compact.get("home") or ""), str(compact.get("away") or "")
    people = {}
    try:
        book = getattr(session, "box_score_book", None) or {}
        people = ((book.get("seasons") or {}).get(str(season)) or {}).get("people") or {}
    except Exception:
        people = {}
    broken: List[Dict[str, Any]] = []

    lg_book, fr_book, se_book = recs["league_season_game"], recs["franchise_game"], recs["season_game"]
    vu_book, vus_book = recs["vs_user"], recs["vs_user_season"]

    def _cur(book: Dict[str, Any], stat: str) -> float:
        r = book.get(stat)
        return float(r.get("value") or 0) if isinstance(r, dict) else 0.0

    # Cheapest bar any line must clear to touch any book (one dict read per slot).
    bars = {
        stat: min(_cur(lg_book, stat), _cur(se_book, stat), _cur(vus_book, stat) if stat != "saves" else 1e9)
        for stat in ("goals", "assists", "points", "saves")
    }
    for pid, row in (compact.get("lines") or {}).items():
        try:
            g, a = int(row[_REC_G] or 0), int(row[_REC_A] or 0)
            saves = int(row[_REC_SAVES] or 0) if int(row[_REC_SA] or 0) > 0 else 0
        except (IndexError, TypeError, ValueError):
            continue
        vals = (("goals", g), ("assists", a), ("points", g + a), ("saves", saves))
        if not any(v > 0 and v > bars[stat] for stat, v in vals):
            continue
        tid = str(row[0])
        opp = aid if tid == hid else hid
        name = str((people.get(pid) or ["Player"])[0])

        def _mk(v: float, _pid: str = pid, _name: str = name, _tid: str = tid, _opp: str = opp) -> Any:
            return lambda: _rec_row(v, compact, player_id=_pid, name=_name, team_id=_tid, opp_id=_opp)

        for stat, v in vals:
            if v <= 0 or v <= bars[stat]:
                continue
            _rec_try(lg_book, stat, v, _mk(v))
            if tid == utid:
                if _rec_try(fr_book, stat, v, _mk(v)):
                    broken.append({"book": "franchise_game", "stat": stat, **fr_book[stat]})
                elif _rec_try(se_book, stat, v, _mk(v)):
                    broken.append({"book": "season_game", "stat": stat, **se_book[stat]})
                else:
                    continue
                # A franchise record is also this season's best.
                _rec_try(se_book, stat, v, _mk(v))
            elif utid and opp == utid and stat != "saves":
                if _rec_try(vu_book, stat, v, _mk(v)):
                    broken.append({"book": "vs_user", "stat": stat, **vu_book[stat]})
                    _rec_try(vus_book, stat, v, _mk(v))
                elif _rec_try(vus_book, stat, v, _mk(v)):
                    broken.append({"book": "vs_user_season", "stat": stat, **vus_book[stat]})

    if utid in (hid, aid):
        idx = 0 if utid == hid else 1
        gf = int(compact.get("hg" if idx == 0 else "ag") or 0)
        ga = int(compact.get("ag" if idx == 0 else "hg") or 0)
        shots = sum(list((compact.get("shots") or [[], []])[idx] or []))
        opp = aid if idx == 0 else hid
        for stat, v in (("team_goals", gf), ("team_shots", shots), ("margin", gf - ga)):
            mk = (lambda _v=v: _rec_row(_v, compact, team_id=utid, opp_id=opp, final=f"{gf}-{ga}"))
            if _rec_try(recs["team_alltime"], stat, v, mk):
                broken.append({"book": "team_alltime", "stat": stat, **recs["team_alltime"][stat]})
                _rec_try(recs["team_season"], stat, v, mk)
            elif _rec_try(recs["team_season"], stat, v, mk):
                broken.append({"book": "team_season", "stat": stat, **recs["team_season"][stat]})

    if publish:
        for rec in broken:
            if float(rec.get("value") or 0) < _REC_PUBLISH_MIN.get(str(rec.get("stat")), 99):
                continue
            try:
                _publish_record(session, rec)
            except Exception:
                continue
    return broken


def _publish_record(session: Any, rec: Dict[str, Any]) -> None:
    from app.sim_engine.franchise.storyline_coverage import _emit_public  # noqa: WPS433
    from app.sim_engine.franchise.storyline_engine import _can_fire, _mark_fired, _team_display, _u_current_meta  # noqa: WPS433

    day, _iso, season = _u_current_meta(session)
    book, stat = str(rec.get("book")), str(rec.get("stat"))
    value = int(rec.get("value") or 0)
    label = _REC_LABEL.get(stat, stat)
    name = str(rec.get("name") or "")
    tname = _team_display(session, str(rec.get("team_id") or ""))
    oname = _team_display(session, str(rec.get("opp_id") or ""))
    if book == "franchise_game":
        headline = f"{name} sets a franchise-era mark: {value} {label} in one game"
        summary = f"No {tname} player has posted more {label} in a single game since you took over. It came against {oname}."
        cause, heat = "FRANCHISE_GAME_RECORD", 80
    elif book == "season_game":
        headline = f"{name}'s {value} {label} are the club's single-game high this season"
        summary = f"{name} topped every {tname} single-game {label} total so far this season against {oname}."
        cause, heat = "SEASON_GAME_RECORD", 66
    elif book == "vs_user":
        headline = f"{name} posts the most {label} ever against {oname}"
        summary = f"{name} of {tname} put up {value} {label} against your club, the most any opponent has managed in one game on your watch."
        cause, heat = "OPPONENT_RECORD_VS_USER", 74
    elif book == "vs_user_season":
        headline = f"{name} burns {oname} for {value} {label}"
        summary = f"That is the most {label} an opponent has posted against your club in one game this season."
        cause, heat = "OPPONENT_SEASON_HIGH_VS_USER", 62
    elif book == "team_alltime":
        tname = _team_display(session, str(rec.get("team_id") or ""))
        headline = f"{tname} set a franchise-era single-game high: {value} {label}"
        summary = f"Final {rec.get('final')} against {oname}. The most {label} in one game since you took over."
        cause, heat = "TEAM_GAME_RECORD", 72
    else:
        tname = _team_display(session, str(rec.get("team_id") or ""))
        headline = f"Season high for {tname}: {value} {label}"
        summary = f"Final {rec.get('final')} against {oname}, the club's best single-game number this season."
        cause, heat = "TEAM_SEASON_GAME_HIGH", 58
    stable = f"boxrec|{book}|{stat}|{rec.get('game_id')}"
    if not _can_fire(session, stable, day, "minor")[0]:
        return
    _emit_public(
        session,
        headline=headline,
        summary=summary,
        cause_type=cause,
        category="performance",
        heat=heat,
        team_id=str(rec.get("team_id") or ""),
        player_id=str(rec.get("player_id") or ""),
        player_name=name,
        evidence={"record_book": book, "stat": stat, "value": value, "game_id": rec.get("game_id"), "season": season},
        reporter={"name": "Derek Knox", "outlet": "NBN", "specialty": "performance"},
        stable_key=stable,
    )
    _mark_fired(session, stable, day, "minor", 0)


# ---------------------------------------------------------------------------
# World Juniors → Storylines tab
# ---------------------------------------------------------------------------

def emit_wjc_storylines(
    session: Any,
    prospects: List[Dict[str, Any]],
    *,
    season_sy: int,
    medals: Optional[Dict[str, Any]] = None,
) -> int:
    """Publish the WJC medal result, the tournament scoring leader and every user-org prospect's line."""
    from app.sim_engine.franchise.storyline_coverage import _emit_public  # noqa: WPS433
    from app.sim_engine.franchise.storyline_engine import _can_fire, _mark_fired, _u_current_meta  # noqa: WPS433

    day, _iso, _season = _u_current_meta(session)
    bundle = getattr(session, "wjc_tournament_bundle", None) or {}
    labels = dict(bundle.get("medal_labels") or {})
    medals = dict(medals or bundle.get("medals") or {})
    reporter = {"name": "Alex Petrov", "outlet": "Future Ice", "specialty": "draft"}
    emitted = 0

    def _fire(stable: str, **kw: Any) -> None:
        nonlocal emitted
        if not _can_fire(session, stable, day, "minor")[0]:
            return
        _emit_public(session, reporter=reporter, stable_key=stable, category="prospects", **kw)
        _mark_fired(session, stable, day, "minor", 0)
        emitted += 1

    gold = labels.get("gold") or medals.get("gold")
    if gold:
        gg = ((bundle.get("playoffs") or {}).get("gold")) or {}
        score = ""
        if isinstance(gg, dict) and gg.get("home_goals") is not None:
            hi, lo = sorted((int(gg.get("home_goals") or 0), int(gg.get("away_goals") or 0)), reverse=True)
            score = f" {hi}-{lo}"
        silver = labels.get("silver") or medals.get("silver") or "the runner-up"
        bronze = labels.get("bronze") or medals.get("bronze")
        _fire(
            f"wjc|medals|{season_sy}",
            headline=f"{gold} wins World Junior gold",
            summary=f"{gold} beat {silver}{score} in the gold-medal game." + (f" {bronze} took bronze." if bronze else ""),
            cause_type="WJC_MEDAL_RESULT",
            heat=70,
            evidence={"year": season_sy, "gold": gold, "silver": silver, "bronze": bronze},
        )

    played = [p for p in prospects or [] if isinstance(p, dict) and int(p.get("tournament_gp") or 0) > 0 and not p.get("is_npc")]
    skaters = [p for p in played if str(p.get("position") or "").upper() != "G"]
    if skaters:
        top = max(skaters, key=lambda p: (int(p.get("tournament_pts") or 0), int(p.get("tournament_g") or 0)))
        pts, g = int(top.get("tournament_pts") or 0), int(top.get("tournament_g") or 0)
        nm = str(top.get("name") or top.get("player_name") or "A prospect")
        _fire(
            f"wjc|leader|{season_sy}",
            headline=f"{nm} leads the World Juniors with {pts} points",
            summary=f"{nm} ({top.get('wjc_country_label') or ''}) finished with {g} goals and {pts - g} assists in {int(top.get('tournament_gp') or 0)} games.",
            cause_type="WJC_SCORING_LEADER",
            heat=60,
            player_id=str(top.get("player_id") or ""),
            player_name=nm,
            evidence={"year": season_sy, "points": pts, "goals": g},
        )

    utid = str(getattr(session, "user_team_id", "") or "")
    for p in played:
        mine = bool(p.get("is_user_prospect")) or str(p.get("prospect_classification") or "") == "drafted_user"
        if not mine:
            continue
        nm = str(p.get("name") or p.get("player_name") or "Prospect")
        gp = int(p.get("tournament_gp") or 0)
        country = str(p.get("wjc_country_label") or "")
        if str(p.get("position") or "").upper() == "G":
            sv = p.get("tournament_sv_pct")
            line = f"{int(p.get('tournament_w') or 0)}-{int(p.get('tournament_l') or 0)}" + (f", {float(sv):.3f} SV%" if sv else "")
        else:
            g, pts = int(p.get("tournament_g") or 0), int(p.get("tournament_pts") or 0)
            line = f"{g}G {pts - g}A in {gp} GP"
        _fire(
            f"wjc|prospect|{season_sy}|{p.get('player_id') or p.get('draft_prospect_id')}",
            headline=f"Your prospect {nm} at the World Juniors: {line}",
            summary=f"{nm} played {gp} games for {country}." + (f" Draft stock moved {int(p.get('stock_delta') or 0):+d}." if int(p.get("stock_delta") or 0) else ""),
            cause_type="WJC_USER_PROSPECT",
            heat=56,
            team_id=utid,
            player_id=str(p.get("player_id") or ""),
            player_name=nm,
            evidence={"year": season_sy, "country": country, "line": line},
        )
        if emitted >= 12:
            break
    return emitted
