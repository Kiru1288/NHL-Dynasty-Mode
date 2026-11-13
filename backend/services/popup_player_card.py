"""Player card for alert popups (waivers, storylines, injuries, legal, meetings).

The popup used to carry only a name, OVR and POT. This adds what a GM wants at a glance: age, contract,
this season's line, league percentiles against players at his position, and his trade value.
Everything is read from the session; a value that isn't known is left out, never invented.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_POP_KINDS_WITH_PLAYER = {"waiver", "storyline", "legal_trouble", "player_meeting", "breaking_news", "injury", "fa_decision"}


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _group(pos: str) -> str:
    p = str(pos or "").upper()
    if p.startswith("G"):
        return "G"
    if p in ("D", "LD", "RD") or p.endswith("D") and len(p) <= 2:
        return "D"
    return "F"


def _pct_rank(values: List[float], v: float, higher_better: bool = True) -> Optional[int]:
    if len(values) < 8:
        return None
    below = sum(1 for x in values if (x < v if higher_better else x > v))
    equal = sum(1 for x in values if x == v)
    return int(round(100.0 * (below + 0.5 * equal) / len(values)))


def _metrics(row: Dict[str, Any], grp: str) -> Dict[str, float]:
    gp = max(1, int(_f(row.get("gp"))))
    toi_min = _f(row.get("toi_sec")) / 60.0
    out: Dict[str, float] = {}
    if grp == "G":
        sa = _f(row.get("sa", row.get("shots_against")))
        sv = _f(row.get("saves", row.get("sv")))
        ga = _f(row.get("ga"))
        if sa > 0:
            out["sv_pct"] = sv / sa
        mins = toi_min if toi_min >= gp * 30 else gp * 60.0
        out["gaa"] = ga * 60.0 / max(1.0, mins)
        out["win_pct"] = _f(row.get("w")) / gp
        return out
    g, a = _f(row.get("g")), _f(row.get("a"))
    pts = _f(row.get("pts"), g + a)
    out["pts_gp"] = pts / gp
    out["g_gp"] = g / gp
    if toi_min > gp * 5:
        out["pts60"] = pts * 60.0 / toi_min
        out["toi_gp"] = toi_min / gp
    sog = _f(row.get("sog", row.get("shots")))
    if sog > 0:
        out["sh_pct"] = g / sog
        out["sog_gp"] = sog / gp
    xgf, xga = _f(row.get("xgf")), _f(row.get("xga"))
    if xgf + xga > 0.5:
        out["xgf_pct"] = xgf / (xgf + xga)
    if row.get("plus_minus") is not None:
        out["plus_minus"] = _f(row.get("plus_minus"))
    hb = _f(row.get("hit")) + _f(row.get("blk"))
    out["phys_gp"] = hb / gp
    return out


_LABELS = {
    "pts_gp": ("Points / GP", True, 2), "g_gp": ("Goals / GP", True, 2), "pts60": ("Points / 60", True, 2),
    "toi_gp": ("TOI / GP", True, 1), "sh_pct": ("Shooting %", True, 3), "sog_gp": ("Shots / GP", True, 1),
    "xgf_pct": ("xGF%", True, 3), "plus_minus": ("+/-", True, 0), "phys_gp": ("Hits + blocks / GP", True, 1),
    "sv_pct": ("Save %", True, 3), "gaa": ("GAA", False, 2), "win_pct": ("Win %", True, 3),
}
_ORDER = {"F": ["pts_gp", "g_gp", "pts60", "xgf_pct", "sog_gp", "sh_pct", "toi_gp", "phys_gp"],
          "D": ["pts_gp", "xgf_pct", "toi_gp", "plus_minus", "phys_gp", "pts60", "sog_gp"],
          "G": ["sv_pct", "gaa", "win_pct"]}


def build_player_card(session: Any, player_id: str, team_id: str = "") -> Optional[Dict[str, Any]]:
    errs: List[str] = []
    try:
        card = _build_player_card(session, player_id, team_id, errs)
    except Exception as exc:  # never lose the whole card to one bad field
        logger.warning("popup player card failed for %s", player_id, exc_info=True)
        card = {"player_id": str(player_id or "")}
        errs.append(f"card: {type(exc).__name__}: {exc}")
    if errs and card is not None:
        card["_errors"] = errs[:6]
    return card


def _build_player_card(session: Any, player_id: str, team_id: str, errs: List[str]) -> Optional[Dict[str, Any]]:
    from services import franchise_sim as fs

    pid = str(player_id or "")
    if not pid or session is None:
        return None
    try:
        player = fs._trade_popup_resolve_player(session, str(team_id or ""), pid)
    except Exception as exc:
        player = None
        errs.append(f"resolve: {type(exc).__name__}: {exc}")
    card: Dict[str, Any] = {"player_id": pid}
    league = getattr(getattr(session, "sim", None), "league", None) or getattr(session, "league", None)
    wentry = None
    try:
        for e in list(getattr(league, "waiver_wire", None) or []):
            if isinstance(e, dict) and str(e.get("player_id") or "") == pid:
                wentry = e
        if player is None and wentry is not None:
            player = wentry.get("player_ref")
        if player is None:
            # minors, free agents: anywhere the league still holds him
            for tm in (getattr(session, "team_by_id", None) or {}).values():
                for p in list(getattr(tm, "ahl_roster", None) or []) + list(getattr(tm, "minor_roster", None) or []):
                    if str(getattr(p, "id", "") or "") == pid:
                        player = p
                        break
                if player is not None:
                    break
        if player is None:
            # the league's own team objects (the session map can hold different copies after a reload)
            for tm in list(getattr(league, "teams", None) or []):
                for p in list(getattr(tm, "roster", None) or []) + list(getattr(tm, "ahl_roster", None) or []):
                    if str(getattr(p, "id", "") or getattr(p, "player_id", "") or "") == pid:
                        player = p
                        break
                if player is not None:
                    break
        if player is None:
            for attr in ("players_by_id", "player_by_id", "all_players_by_id"):
                m = getattr(league, attr, None)
                if isinstance(m, dict) and m.get(pid) is not None:
                    player = m.get(pid)
                    break
        if player is None:
            for p in list(getattr(league, "free_agents", None) or []):
                if str(getattr(p, "id", "") or "") == pid:
                    player = p
                    break
    except Exception as exc:
        errs.append(f"lookup: {type(exc).__name__}: {exc}")
    if player is None:
        errs.append("lookup: waiver entry has no player object" if wentry is not None else "lookup: player not found in rosters, minors, waivers or free agents")
    if wentry is not None and not team_id:
        team_id = str(wentry.get("original_team_id") or "")
    pos = ""
    if player is not None:
        pos = fs._pos_str(player)
        card["position"] = pos
        try:
            card["age"] = fs._trade_popup_age(player)
        except Exception:
            pass
        try:
            card["archetype"] = fs._trade_popup_archetype_label(player)
        except Exception:
            pass
        try:
            cap, yrs = fs._trade_popup_contract_bits(player)
            if cap is not None and float(cap) > 0:
                card["cap_hit_m"] = round(float(cap), 2)
            if yrs is not None:
                card["years_left"] = int(yrs)
        except Exception:
            pass
        for attr, key in (("shoots", "shoots"), ("handedness", "shoots"), ("nationality", "country"), ("country", "country")):
            v = getattr(player, attr, None) or getattr(getattr(player, "identity", None), attr, None)
            if v and key not in card:
                txt = str(getattr(v, "value", v))
                if key == "shoots":
                    txt = txt.split(".")[-1].strip()[:1].upper()
                    if txt not in ("L", "R"):
                        continue
                card[key] = txt
    if player is not None:
        try:
            ga = player.group_averages() if callable(getattr(player, "group_averages", None)) else None
            if isinstance(ga, dict) and ga:
                card["skills"] = [{"key": k, "value": int(round(float(v)))} for k, v in ga.items() if v is not None][:8]
        except Exception:
            pass
    if player is not None:
        try:
            career = getattr(player, "career_stats", None)
            rows = (career.get("seasons") or career.get("by_season") or career.get("history")) if isinstance(career, dict) else None
            nhl = [r for r in (rows or []) if isinstance(r, dict) and str(r.get("league") or r.get("league_code") or "NHL").upper() == "NHL" and _f(r.get("gp")) > 0]
            if nhl:
                tot = {"seasons": len(nhl), "gp": int(sum(_f(r.get("gp")) for r in nhl))}
                if str(card.get("position") or "").upper().startswith("G"):
                    tot["w"] = int(sum(_f(r.get("w", r.get("wins"))) for r in nhl))
                    tot["so"] = int(sum(_f(r.get("so", r.get("shutouts"))) for r in nhl))
                else:
                    tot["g"] = int(sum(_f(r.get("g", r.get("goals"))) for r in nhl))
                    tot["a"] = int(sum(_f(r.get("a", r.get("assists"))) for r in nhl))
                    tot["pts"] = int(sum(_f(r.get("pts", r.get("points")), _f(r.get("g", r.get("goals"))) + _f(r.get("a", r.get("assists")))) for r in nhl))
                card["career"] = tot
        except Exception as exc:
            errs.append(f"career: {type(exc).__name__}: {exc}")
    try:
        cur = fs._trade_popup_season_stats(session, pid, player)
        card["season"] = cur
        # last season too when this one is young (or he hasn't played yet)
        if player is not None and (not isinstance(cur, dict) or cur.get("source") != "nhl_prior") and int(_f((cur or {}).get("gp"))) < 20:
            prior = fs._trade_popup_prior_nhl_line(player)
            if prior and int(_f(prior.get("gp"))) > 0:
                lab = str(prior.get("season") or "Last season")
                last = {"label": f"{lab} NHL", "gp": int(_f(prior.get("gp")))}
                if str(pos or "").upper().startswith("G"):
                    last.update(kind="goalie", w=int(_f(prior.get("w", prior.get("wins")))), l=int(_f(prior.get("l", prior.get("losses")))),
                                sv_pct=prior.get("sv_pct", prior.get("save_pct")), gaa=prior.get("gaa"), so=int(_f(prior.get("so", prior.get("shutouts")))))
                else:
                    g_, a_ = int(_f(prior.get("g", prior.get("goals")))), int(_f(prior.get("a", prior.get("assists"))))
                    last.update(kind="skater", g=g_, a=a_, pts=int(_f(prior.get("pts", prior.get("points")), g_ + a_)),
                                plus_minus=prior.get("plus_minus", prior.get("pm")))
                card["last_season"] = last
    except Exception as exc:
        errs.append(f"stats: {type(exc).__name__}: {exc}")
    # percentiles against players at his position with a real sample
    try:
        _percentiles(card, session, pid, pos)
    except Exception as exc:
        errs.append(f"percentiles: {type(exc).__name__}: {exc}")
    if "percentiles" not in card and player is not None and (card.get("last_season") or (card.get("season") or {}).get("source") == "nhl_prior"):
        try:
            _prior_percentiles(card, session, fs, player, pos)
        except Exception as exc:
            errs.append(f"last-season percentiles: {type(exc).__name__}: {exc}")
    try:
        _club(card, session, fs, pid, team_id)
    except Exception as exc:
        errs.append(f"club: {type(exc).__name__}: {exc}")
    return _tail(card, session, fs, pid, pos, player, team_id, league, wentry, errs)


def _prior_row(line: Dict[str, Any]) -> Dict[str, Any]:
    g, a = _f(line.get("g", line.get("goals"))), _f(line.get("a", line.get("assists")))
    return {"gp": _f(line.get("gp")), "g": g, "a": a, "pts": _f(line.get("pts", line.get("points")), g + a),
            "sog": _f(line.get("sog", line.get("shots"))), "plus_minus": line.get("plus_minus", line.get("pm")),
            "toi_sec": _f(line.get("toi_sec")) or _f(line.get("toi_per_game", line.get("atoi"))) * 60.0 * _f(line.get("gp")),
            "w": _f(line.get("w", line.get("wins"))), "ga": _f(line.get("ga")), "sa": _f(line.get("sa", line.get("shots_against"))),
            "saves": _f(line.get("saves", line.get("sv")))}


def _prior_percentiles(card: Dict[str, Any], session: Any, fs: Any, player: Any, pos: str) -> None:
    """No games yet this season: rank last season's line against everyone else's last season."""
    mine_line = fs._trade_popup_prior_nhl_line(player)
    if not mine_line:
        return
    grp = _group(pos)
    label = str(mine_line.get("season") or "")
    mine = _metrics(_prior_row(mine_line), grp)
    cols: Dict[str, List[float]] = {k: [] for k in mine}
    n = 0
    for tm in (getattr(session, "team_by_id", None) or {}).values():
        for q in getattr(tm, "roster", None) or []:
            if _group(fs._pos_str(q)) != grp:
                continue
            ln = fs._trade_popup_prior_nhl_line(q)
            if not ln or (label and str(ln.get("season") or "") != label) or _f(ln.get("gp")) < 20:
                continue
            n += 1
            m = _metrics(_prior_row(ln), grp)
            for k in cols:
                if k in m:
                    cols[k].append(m[k])
    pcts = []
    for k in _ORDER[grp]:
        if k not in mine:
            continue
        label_k, hb, dig = _LABELS[k]
        pr = _pct_rank(cols.get(k) or [], mine[k], hb)
        if pr is None:
            continue
        val = mine[k]
        shown = f"{val * 100:.1f}%" if k in ("sh_pct", "xgf_pct", "win_pct") else (f"{val:.3f}".lstrip("0") if k == "sv_pct" else (f"{val:+.0f}" if k == "plus_minus" else f"{val:.{dig}f}"))
        pcts.append({"key": k, "label": label_k, "value": shown, "pct": pr})
    if pcts:
        card["percentiles"] = pcts[:6]
        card["percentile_pool"] = {"group": {"F": "forwards", "D": "defencemen", "G": "goalies"}[grp], "n": n, "min_gp": 20, "season": label or "last season"}


def _percentiles(card: Dict[str, Any], session: Any, pid: str, pos: str) -> None:
    pss = getattr(session, "player_season_stats", None) or {}
    row = dict(pss.get(pid) or {})
    grp = _group(pos or row.get("position") or row.get("pos") or "")
    gp = int(_f(row.get("gp")))
    if row and gp > 0:
        mine = _metrics(row, grp)
        # the qualifying bar follows the league, not him: early in a season nobody has ten games
        gps = sorted(int(_f(r.get("gp"))) for r in pss.values() if isinstance(r, dict) and int(_f(r.get("gp"))) > 0
                     and _group(r.get("position") or r.get("pos") or "") == grp)
        med = gps[len(gps) // 2] if gps else 1
        min_gp = max(1, min(10, med // 2))
        pool = [r for r in pss.values() if isinstance(r, dict) and _group(r.get("position") or r.get("pos") or "") == grp and int(_f(r.get("gp"))) >= min_gp]
        cols = {k: [] for k in mine}
        for r in pool:
            m = _metrics(r, grp)
            for k in cols:
                if k in m:
                    cols[k].append(m[k])
        pcts = []
        for k in _ORDER[grp]:
            if k not in mine:
                continue
            label, hb, dig = _LABELS[k]
            pr = _pct_rank(cols.get(k) or [], mine[k], hb)
            if pr is None:
                continue
            val = mine[k]
            shown = f"{val * 100:.1f}%" if k in ("sh_pct", "xgf_pct", "win_pct") else (f"{val:.3f}".lstrip("0") if k == "sv_pct" else (f"{val:+.0f}" if k == "plus_minus" else (f"{val:.{dig}f}")))
            pcts.append({"key": k, "label": label, "value": shown, "pct": pr})
        if pcts:
            card["percentiles"] = pcts[:6]
            card["percentile_pool"] = {"group": {"F": "forwards", "D": "defencemen", "G": "goalies"}[grp], "n": len(pool), "min_gp": min_gp, "season": "this season"}


def _club(card: Dict[str, Any], session: Any, fs: Any, pid: str, team_id: str) -> None:
    tb = getattr(session, "team_by_id", None) or {}
    src = tb.get(str(team_id)) if team_id and str(team_id) in tb and any(str(getattr(p, "id", "")) == pid for p in (getattr(tb.get(str(team_id)), "roster", None) or [])) else None
    if src is None:
        for tm in tb.values():
            if any(str(getattr(p, "id", "")) == pid for p in (getattr(tm, "roster", None) or [])):
                src = tm
                break
    club = src if src is not None else tb.get(str(team_id or ""))
    if club is not None:
        cid = str(getattr(club, "team_id", "") or getattr(club, "id", "") or team_id or "")
        card["team_id"] = cid
        try:
            card["team_abbr"] = fs._trade_popup_team_abbr(session, cid)
        except Exception:
            pass
        city = str(getattr(club, "city", "") or getattr(club, "market", "") or "").strip()
        nick = str(getattr(club, "name", "") or getattr(club, "nickname", "") or "").strip()
        full = str(getattr(club, "full_name", "") or "").strip() or (f"{city} {nick}".strip() if nick and city and city not in nick else nick)
        if full:
            card["team_name"] = full


def _tail(card: Dict[str, Any], session: Any, fs: Any, pid: str, pos: str, player: Any, team_id: str, league: Any, wentry: Any, errs: List[str]) -> Dict[str, Any]:
    tb = getattr(session, "team_by_id", None) or {}
    src = None
    for tm in tb.values():
        if any(str(getattr(p, "id", "")) == pid for p in (getattr(tm, "roster", None) or [])):
            src = tm
            break
    # waivers: when claims close, where the user sits in the claim order, and whether he fits under the cap
    if wentry is not None:
        try:
            from services import waivers as wv

            now = int(getattr(session, "calendar_cursor", 0) or 0)
            exp = int(wentry.get("expires_cursor") or now)
            info: Dict[str, Any] = {"days_left": max(0, exp - now), "reason": str(wentry.get("reason") or "").replace("_", " ")}
            if wentry.get("cap_hit_m") is not None and "cap_hit_m" not in card and float(wentry.get("cap_hit_m") or 0) > 0:
                card["cap_hit_m"] = round(float(wentry["cap_hit_m"]), 2)
            if wentry.get("years_remaining") is not None and "years_left" not in card:
                card["years_left"] = int(wentry["years_remaining"])
            teams = {str(getattr(t, "team_id", "")): t for t in list(getattr(league, "teams", None) or [])}
            order = sorted([t for k, t in teams.items() if k != str(wentry.get("original_team_id"))], key=lambda t: wv._waiver_priority_key(session, t))
            utid = str(getattr(session, "user_team_id", "") or "")
            pos_u = next((i + 1 for i, t in enumerate(order) if str(getattr(t, "team_id", "")) == utid), None)
            if pos_u:
                info["claim_priority"] = pos_u
                info["claim_teams"] = len(order)
            ut = teams.get(utid)
            if ut is not None:
                try:
                    from services.contract_economy import get_team_cap_snapshot_full

                    snap = get_team_cap_snapshot_full(ut, league)
                    space = snap.get("usable_cap_space_m", snap.get("cap_space"))
                    if space is not None:
                        info["user_cap_space_m"] = round(float(space), 2)
                        info["fits_cap"] = float(space) >= float(wentry.get("cap_hit_m") or 0)
                except Exception:
                    pass
            card["waiver"] = info
        except Exception as exc:
            errs.append(f"waiver info: {type(exc).__name__}: {exc}")
    # where he'd slot on the user's depth chart at his position
    try:
        utm = (getattr(session, "team_by_id", None) or {}).get(str(getattr(session, "user_team_id", "") or ""))
        if utm is not None and player is not None and str(card.get("team_id") or "") != str(getattr(session, "user_team_id", "")):
            grp_u = _group(pos)
            mine_ovr = float(fs._trade_popup_ovr(player) or 0)
            same = sorted((float(fs._trade_popup_ovr(q) or 0) for q in (getattr(utm, "roster", None) or []) if _group(fs._pos_str(q)) == grp_u), reverse=True)
            if same and mine_ovr > 0:
                rank = 1 + sum(1 for x in same if x > mine_ovr)
                card["depth_fit"] = {"rank": rank, "of": len(same) + 1, "group": {"F": "forwards", "D": "defencemen", "G": "goalies"}[grp_u]}
    except Exception as exc:
        errs.append(f"depth fit: {type(exc).__name__}: {exc}")
    # trade value as the user's club would see it
    if player is not None:
        try:
            from app.sim_engine.trades.trade_value import evaluate_player_asset_value

            acq = tb.get(str(getattr(session, "user_team_id", "") or ""))
            val = evaluate_player_asset_value(player, src, acq, league, context={"read_only": True})
            if isinstance(val, dict) and val.get("total") is not None:
                card["trade_value"] = round(float(val.get("total")), 1)
                if val.get("value_tier"):
                    card["value_tier"] = str(val.get("value_tier"))
        except Exception as exc:
            errs.append(f"trade value: {type(exc).__name__}: {exc}")
    return card


def enrich_popups(session: Any, pops: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    """Attach a player card to the first few popups that name a player (cached on the popup)."""
    out = []
    n = 0
    for p in pops:
        if isinstance(p, dict) and n < limit and p.get("player_id") and not (p.get("player_card") or {}).get("age") and str(p.get("kind") or "") in _POP_KINDS_WITH_PLAYER:
            n += 1
            try:
                card = build_player_card(session, str(p.get("player_id")), str(p.get("team_id") or ""))
            except Exception:
                logger.debug("popup card failed", exc_info=True)
                card = None
            p["player_card"] = card or {}
        out.append(p)
    return out
