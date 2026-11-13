"""Training camp opening: the bridge from the end of free agency into the new season.

Built once when the next season is generated (stage ``next_season_reveal``) and cached on
``session.next_season_payload["training_camp"]``. Everything comes from the save: last season's
archived record, the user club's NHL + AHL players, contracts, the new schedule and calendar.
Nothing is invented: if a fact is missing it is left out.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# NHL dressing lineup per position group, and how many places near the cut line count as a battle.
LINEUP_SLOTS = {"F": 12, "D": 6, "G": 2}
SIGNING_SOURCES = {"free_agent", "signed", "offer_sheet", "professional_tryout", "user", "contract_desk", "qualifying_offer"}


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _group(pos: str) -> str:
    p = str(pos or "").upper()
    if p.startswith("G"):
        return "G"
    if p in ("D", "LD", "RD") or p.startswith("D"):
        return "D"
    return "F"


def _ovr(player: Any) -> float:
    try:
        from services.franchise_sim import _player_display_ovr99

        return round(float(_player_display_ovr99(player)), 1)
    except Exception:
        fn = getattr(player, "ovr", None)
        v = _f(fn() if callable(fn) else fn)
        return round(v * 99.0 if v <= 1.5 else v, 1)


def _age(player: Any) -> Optional[int]:
    try:
        from services.franchise_sim import _player_age_int

        a = int(_player_age_int(player))
        return a or None
    except Exception:
        a = getattr(player, "age", None)
        return int(a) if isinstance(a, (int, float)) and a else None


def _contract(player: Any) -> Dict[str, Any]:
    c = getattr(player, "contract", None)
    if isinstance(c, dict):
        return c
    if c is not None and hasattr(c, "__dict__"):
        return dict(vars(c))
    return {}


def _cap_hit(player: Any) -> float:
    try:
        from services.contract_economy import _contract_cap_hit_m

        return round(float(_contract_cap_hit_m(player)), 3)
    except Exception:
        c = _contract(player)
        return round(_f(c.get("cap_hit", c.get("aav"))), 3)


def _years_left(player: Any) -> Optional[int]:
    try:
        from services.contract_economy import _contract_years_remaining

        return int(_contract_years_remaining(player))
    except Exception:
        c = _contract(player)
        y = c.get("years_remaining", c.get("years"))
        return int(y) if isinstance(y, (int, float)) else None


def _arrival(player: Any, offseason_year: int) -> Optional[str]:
    """How a player joined the club this offseason, if he did."""
    if getattr(player, "acquired_via_trade", False) and int(_f(getattr(player, "acquired_via_trade_season", 0))) in (offseason_year, offseason_year + 1):
        return "Trade"
    c = _contract(player)
    src = str(c.get("source") or "").lower()
    signed = int(_f(c.get("signed_year"), 0))
    if src in SIGNING_SOURCES and signed in (offseason_year, offseason_year + 1):
        return "Tryout" if src == "professional_tryout" else "Free agency"
    draft_year = int(_f(getattr(player, "draft_year", 0)))
    if draft_year == offseason_year + 1 or draft_year == offseason_year:
        if src == "elc" or c.get("is_entry_level"):
            return "Draft"
    return None


def _row(player: Any, level: str, offseason_year: int) -> Dict[str, Any]:
    pid = str(getattr(player, "id", None) or getattr(player, "player_id", "") or "")
    raw_pos = getattr(player, "position", "") or ""
    pos = str(getattr(raw_pos, "value", raw_pos) or "")
    c = _contract(player)
    row = {
        "player_id": pid,
        "name": str(getattr(player, "name", None) or getattr(player, "full_name", "") or pid),
        "position": pos,
        "group": _group(pos),
        "age": _age(player),
        "ovr": _ovr(player),
        "level": level,
        "cap_hit": _cap_hit(player),
        "years_left": _years_left(player),
        "entry_level": bool(c.get("is_entry_level")),
        "arrival": _arrival(player, offseason_year),
    }
    delta = getattr(player, "season_ovr_delta", None)
    if isinstance(delta, (int, float)):
        row["summer_delta"] = round(float(delta), 1)
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row, nhl_headshot_fields_from_player

        row = merge_headshot_into_row(row, player)
        row.update({k: v for k, v in (nhl_headshot_fields_from_player(player) or {}).items() if v})
    except Exception:
        pass
    if bool(getattr(player, "is_captain", False)) or str(getattr(player, "captaincy_status", "") or "").lower() in ("c", "captain"):
        row["captain"] = True
    return row


def _battles(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Locks, open battles near the cut line, and long shots, per position group."""
    out: Dict[str, Any] = {"groups": [], "battles": []}
    labels = {"F": "Forwards", "D": "Defence", "G": "Goalies"}
    spot_name = {"F": "forward", "D": "defence", "G": "goalie"}
    for g, slots in LINEUP_SLOTS.items():
        rows = sorted([r for r in pool if r["group"] == g], key=lambda r: (-r["ovr"], r["level"] != "NHL"))
        if not rows:
            continue
        cut = rows[min(slots, len(rows)) - 1]["ovr"]
        lock_n = max(0, slots - (2 if g != "G" else 1))
        locks = rows[:lock_n]
        contenders = []
        for r in rows[lock_n:]:
            if len(contenders) >= (4 if g != "G" else 3):
                break
            if r["ovr"] >= cut - 6:
                contenders.append({**r, "gap": round(r["ovr"] - cut, 1)})
        open_spots = slots - lock_n
        extra = rows[lock_n + len(contenders):]
        out["groups"].append({
            "group": g,
            "label": labels[g],
            "slots": slots,
            "in_camp": len(rows),
            "locks": locks,
            "cut_ovr": cut,
            "long_shots": extra[:6],
        })
        if len(contenders) > open_spots:
            last = "spot" if open_spots == 1 else "spots"
            out["battles"].append({
                "group": g,
                "title": f"Last {open_spots} {spot_name[g]} {last}" if g != "G" else "The backup job",
                "open_spots": open_spots,
                "candidates": contenders,
                "closeness": round(contenders[0]["ovr"] - contenders[-1]["ovr"], 1),
            })
    out["battles"].sort(key=lambda b: b["closeness"])
    return out


def _last_season(session: Any, uid: str) -> Dict[str, Any]:
    hist = list(getattr(session, "season_history", None) or [])
    if not hist:
        return {}
    last = hist[-1] if isinstance(hist[-1], dict) else {}
    teams = last.get("teams") or {}
    me = teams.get(uid) or {}
    if not me:
        return {"season_label": last.get("season_label")}
    ranked = sorted(teams.items(), key=lambda kv: -_f(kv[1].get("pts")))
    rank = next((i + 1 for i, (tid, _) in enumerate(ranked) if tid == uid), None)
    champ = str(last.get("champion_id") or "")
    out = {
        "season_label": last.get("season_label"),
        "w": int(_f(me.get("w"))),
        "l": int(_f(me.get("l"))),
        "otl": int(_f(me.get("otl"))),
        "pts": int(_f(me.get("pts"))),
        "gf": int(_f(me.get("gf"))),
        "ga": int(_f(me.get("ga"))),
        "league_rank": rank,
        "teams": len(teams),
        "champion": champ == uid,
        "champion_abbr": (teams.get(champ) or {}).get("abbr") if champ else None,
    }
    for k in ("xgf_pct", "pdo", "luck_label"):
        if me.get(k) is not None:
            out[k] = me.get(k)
    return out


def _expectation(last: Dict[str, Any]) -> Dict[str, str]:
    if not last or last.get("pts") is None:
        return {"tone": "neutral", "title": "A fresh start", "line": "No record from last season on file."}
    rank, n = last.get("league_rank") or 0, last.get("teams") or 32
    if last.get("champion"):
        return {"tone": "gold", "title": "Defend the Cup", "line": "The banner is going up. Every opponent circles your date on the calendar."}
    if rank and rank <= max(4, n // 8):
        return {"tone": "high", "title": "Contend", "line": f"Finished {rank} of {n} in the standings. Anything short of a long run will sting."}
    if rank and rank <= n // 2:
        return {"tone": "mid", "title": "Take the next step", "line": f"Finished {rank} of {n}. The room expects a playoff spot and a real push."}
    if rank and rank <= (3 * n) // 4:
        return {"tone": "mid", "title": "Get back in the race", "line": f"Finished {rank} of {n}. Fans want meaningful games after Christmas."}
    return {"tone": "low", "title": "Show progress", "line": f"Finished {rank} of {n}. Development and effort will be judged as much as wins."}


def build_training_camp(session: Any) -> Dict[str, Any]:
    uid = str(getattr(session, "user_team_id", "") or "")
    team = (getattr(session, "team_by_id", None) or {}).get(uid)
    if team is None:
        return {}
    ns = getattr(session, "next_season_payload", None) or {}
    season_year = int(_f(ns.get("season_year"), _f(getattr(session, "season_calendar_year", 0)) + 1))
    offseason_year = season_year - 1

    nhl = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    ahl = [p for p in (getattr(team, "ahl_roster", None) or []) if not getattr(p, "retired", False)]
    rows = [_row(p, "NHL", offseason_year) for p in nhl] + [_row(p, "AHL", offseason_year) for p in ahl]
    # Camp invites are the top of the AHL depth chart plus anyone new; deep minor leaguers stay home.
    nhl_rows = [r for r in rows if r["level"] == "NHL"]
    ahl_rows = sorted([r for r in rows if r["level"] == "AHL"], key=lambda r: -r["ovr"])
    invited = nhl_rows + [r for r in ahl_rows if r["arrival"]] + [r for r in ahl_rows if not r["arrival"]][:10]

    new_faces = sorted([r for r in invited if r["arrival"]], key=lambda r: -r["ovr"])
    rookies = sorted([r for r in invited if (r["age"] or 99) <= 21 or r["entry_level"]], key=lambda r: -r["ovr"])
    veterans = sorted([r for r in nhl_rows if (r["age"] or 0) >= 33], key=lambda r: -r["ovr"])
    risers = sorted([r for r in invited if _f(r.get("summer_delta")) >= 2], key=lambda r: -_f(r.get("summer_delta")))
    depth = _battles(invited)

    cap: Dict[str, Any] = {}
    try:
        from services.franchise_sim import _team_cap_snapshot

        snap = _team_cap_snapshot(team, getattr(session, "sim", None), season_year=season_year)
        cap = {"cap": snap.get("salary_cap"), "payroll": snap.get("cap_hit"), "space": snap.get("cap_space")}
    except Exception:
        log.debug("camp cap snapshot failed", exc_info=True)

    last = _last_season(session, uid)
    avg_age = round(sum(r["age"] for r in nhl_rows if r["age"]) / max(1, sum(1 for r in nhl_rows if r["age"])), 1) if nhl_rows else None
    avg_ovr = round(sum(r["ovr"] for r in nhl_rows) / max(1, len(nhl_rows)), 1) if nhl_rows else None

    # The road from camp to the playoffs, from the league calendar for this season.
    beats: List[Dict[str, Any]] = []
    wanted = (
        ("preseason_start", "Camp opens", "camp"),
        ("preseason_finale", "Final cuts", "cuts"),
        ("opening_night", "Opening night", "opening"),
        ("thanksgiving", "Thanksgiving check", "check"),
        ("roster_freeze", "Holiday freeze", "freeze"),
        ("trade_deadline", "Trade deadline", "deadline"),
        ("playoffs_start", "Playoffs begin", "playoffs"),
    )
    try:
        from services.nhl_season_calendar import season_anchor_event_markers

        anchors = season_anchor_event_markers(season_year) or []
    except Exception:
        anchors = []
    for suffix, label, kind in wanted:
        hit = next((a for a in anchors if str(a.get("id") or a.get("key") or "").endswith(suffix)), None)
        iso = (hit or {}).get("iso") or (hit or {}).get("date")
        if kind == "camp":
            iso = ns.get("preseason_start") or iso
        if kind == "opening":
            iso = ns.get("opening_night") or iso
        if iso:
            beats.append({"key": kind, "label": label, "iso": str(iso)[:10]})

    captain = next((r for r in nhl_rows if r.get("captain")), None)
    team_name = str(getattr(team, "full_name", None) or getattr(team, "name", None) or f"{getattr(team, 'city', '')} {getattr(team, 'nickname', '')}").strip()
    return {
        "season_year": season_year,
        "season_label": ns.get("season_label") or f"{season_year}–{season_year + 1}",
        "team_name": team_name,
        "team_abbr": str(getattr(team, "abbreviation", None) or getattr(team, "abbr", "") or "")[:3].upper(),
        "first_opponent": ns.get("first_opponent"),
        "opening_night": ns.get("opening_night"),
        "camp_opens": ns.get("preseason_start"),
        "schedule_games": ns.get("schedule_games"),
        "beats": beats,
        "in_camp": len(invited),
        "nhl_count": len(nhl_rows),
        "roster_limit": 23,
        "avg_age": avg_age,
        "avg_ovr": avg_ovr,
        "captain": captain,
        "new_faces": new_faces[:10],
        "new_faces_total": len(new_faces),
        "rookies": rookies[:8],
        "rookies_total": len(rookies),
        "veterans": veterans[:5],
        "summer_risers": risers[:6],
        "depth": depth,
        "cap": cap,
        "last_season": last,
        "expectation": _expectation(last),
    }


def attach_training_camp(session: Any) -> Dict[str, Any]:
    """Return next_season payload with a cached camp block (built once per generated season)."""
    ns = getattr(session, "next_season_payload", None)
    if not isinstance(ns, dict) or not ns:
        return ns or {}
    camp = ns.get("training_camp")
    if not isinstance(camp, dict) or int(_f(camp.get("season_year"))) != int(_f(ns.get("season_year"))):
        try:
            ns["training_camp"] = build_training_camp(session)
        except Exception:
            log.warning("training camp build failed", exc_info=True)
    return ns
