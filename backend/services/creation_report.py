"""League creation report for the setup loading notepad.

``GET /api/franchise/creation-report`` feeds the hand-written notepad shown while a
new franchise is being built. Most of that time is spent *inside* POST
/api/franchise/start, before a session exists, so this module also listens to
``franchise_sim._franchise_startup_stage`` (wrapped below, original still called) and
keeps a light reference to the league being built. Once the league's rosters exist the
report is computed from it; once the session is saved (X-Franchise-Session header) the
report comes from the session and gains contract / cap facts.

All reads are best-effort and never raise: the loading screen falls back to generic
lines when ``ready`` is false.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import APIRouter, Header
except ImportError:  # pragma: no cover - lets the module import in tooling without FastAPI
    APIRouter = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]

import services.franchise_sim as franchise_sim
from services.franchise_store import get_session

log = logging.getLogger(__name__)

router = APIRouter() if APIRouter is not None else None

# ---------------------------------------------------------------------------
# Live capture of the franchise being built
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_LIVE: Dict[str, Any] = {}
_LIVE_CACHE: Dict[str, Any] = {}

# (substring of the raw stage message, friendly notepad line, key)
_STAGE_LINES: List[Tuple[str, str, str]] = [
    ("SimEngine import complete", "Unboxing the sim engine...", "engine"),
    ("team resolution", "Hanging 32 team banners...", "teams"),
    ("preparing real NHL roster", "Importing real NHL rosters...", "real_rosters"),
    ("user team resolved", "Handing you the keys to the office...", "user_team"),
    ("head coach generated", "Hiring your head coach...", "coach"),
    ("generating abstract schedule", "Drafting the schedule...", "schedule"),
    ("building NHL season calendar", "Printing the season calendar...", "calendar"),
    ("map_abstract_schedule_to_calendar", "Booking arenas for every game night...", "arenas"),
    ("_finalize_schedule_after_generation", "Checking back-to-backs and road trips...", "finalize"),
    ("creating FranchiseSession", "Opening the league office...", "session"),
    ("roster contracts bootstrapped", "Signing contracts...", "contracts"),
    ("start_franchise complete", "Filing the paperwork...", "complete"),
]


def _wrap_startup_stage() -> None:
    original = getattr(franchise_sim, "_franchise_startup_stage", None)
    if original is None or getattr(original, "_creation_report_wrapped", False):
        return

    def _stage(msg: str) -> None:
        try:
            _record_stage(str(msg or ""), sys._getframe(1))
        except Exception:
            log.debug("creation report stage capture failed", exc_info=True)
        original(msg)

    _stage._creation_report_wrapped = True  # type: ignore[attr-defined]
    franchise_sim._franchise_startup_stage = _stage


def _record_stage(msg: str, frame: Any) -> None:
    now = time.time()
    loc = dict(getattr(frame, "f_locals", None) or {})
    with _LOCK:
        if msg.startswith("SimEngine import complete"):
            _LIVE.clear()
            _LIVE_CACHE.clear()
            _LIVE["started_at"] = now
            _LIVE["steps"] = []
        steps = _LIVE.setdefault("steps", [])
        for needle, line, key in _STAGE_LINES:
            if needle in msg and not any(s["key"] == key for s in steps):
                steps.append({"key": key, "line": line, "at": now})
                break
        _LIVE["raw_stage"] = msg
        _LIVE["updated_at"] = now
        for name in ("league", "user_team", "season_y", "nhl_cal", "session", "uid"):
            if name in loc and loc[name] is not None:
                _LIVE[name] = loc[name]
        # Rosters are only safe to read once the user team is resolved (real NHL import done).
        if "user_team" in loc and loc.get("user_team") is not None:
            _LIVE["rosters_ready"] = True


_wrap_startup_stage()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _ovr(p: Any) -> float:
    try:
        return float(franchise_sim._player_display_ovr99(p) or 0)
    except Exception:
        try:
            return float(franchise_sim._player_ovr_99_scale(p) or 0)
        except Exception:
            return 0.0


def _age(p: Any) -> int:
    try:
        return int(franchise_sim._player_age_int(p) or 0)
    except Exception:
        return 0


def _cap(p: Any) -> float:
    try:
        return float(franchise_sim._player_cap_hit_millions(p) or 0)
    except Exception:
        return 0.0


def _pot(p: Any) -> Optional[float]:
    r = getattr(p, "ratings", None)
    if isinstance(r, dict) and r.get("dev_potential") is not None:
        try:
            v = float(r["dev_potential"])
            return v * 99.0 if v <= 1.5 else v
        except (TypeError, ValueError):
            return None
    return None


def _name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", None) or getattr(p, "id", "") or "Unknown")


def _pos(p: Any) -> str:
    pos = getattr(p, "position", None) or getattr(getattr(p, "identity", None), "position", None)
    return str(getattr(pos, "value", pos) or "").upper()


def _tid(t: Any) -> str:
    raw = getattr(t, "team_id", None)
    return str(raw if raw is not None else getattr(t, "id", ""))


def _team_name(t: Any) -> str:
    try:
        return franchise_sim._display_team(t)
    except Exception:
        return str(getattr(t, "name", "") or _tid(t))


def _team_abbr(t: Any) -> str:
    ab = getattr(t, "abbreviation", None) or getattr(t, "abbr", None)
    if ab:
        return str(ab).upper()
    try:
        return str(franchise_sim._team_abbr(t, _tid(t)) or "").upper()[:3]
    except Exception:
        return _tid(t)


def _live_players(t: Any) -> List[Any]:
    return [p for p in list(getattr(t, "roster", None) or []) if not getattr(p, "retired", False)]


def _pretty_date(iso: str) -> str:
    try:
        d = date.fromisoformat(str(iso)[:10])
        return d.strftime("%B %d, %Y").replace(" 0", " ")
    except Exception:
        return str(iso or "")


def _player_row(p: Any, t: Any, ovr: float) -> Dict[str, Any]:
    return {
        "name": _name(p),
        "pos": _pos(p),
        "age": _age(p),
        "ovr": int(round(ovr)),
        "team": _team_abbr(t),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _build_report(
    teams: List[Any],
    user_tid: str,
    *,
    season_y: Any = None,
    start_iso: str = "",
    with_cap: bool = False,
) -> Dict[str, Any]:
    team_rows: List[Dict[str, Any]] = []
    all_players: List[Tuple[Any, Any, float]] = []
    prospects: List[Tuple[Any, Any, float, float]] = []

    for t in teams:
        roster = _live_players(t)
        scored = sorted(((p, _ovr(p)) for p in roster), key=lambda x: x[1], reverse=True)
        top = [o for _, o in scored[:18] if o > 0]
        strength = round(sum(top) / len(top), 1) if top else 0.0
        goalies = [(p, o) for p, o in scored if _pos(p) == "G"]
        star = scored[0] if scored else None
        team_rows.append({
            "id": _tid(t),
            "abbr": _team_abbr(t),
            "name": _team_name(t),
            "division": str(getattr(t, "division", "") or ""),
            "conference": str(getattr(t, "conference", "") or ""),
            "ovr": strength,
            "star": _player_row(star[0], t, star[1]) if star else None,
            "goalie": _player_row(goalies[0][0], t, goalies[0][1]) if goalies else None,
            "avg_age": round(sum(_age(p) for p in roster) / len(roster), 1) if roster else 0,
        })
        for p, o in scored:
            all_players.append((p, t, o))
        seen = set()
        for p in roster + list(getattr(t, "prospects", None) or []) + list(getattr(t, "ahl_roster", None) or []):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            age = _age(p)
            pot = _pot(p)
            if pot is None or age <= 0 or age > 21:
                continue
            prospects.append((p, t, pot, _ovr(p)))

    ranked = sorted([r for r in team_rows if r["ovr"] > 0], key=lambda r: r["ovr"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    n = len(ranked)

    best_players = sorted(all_players, key=lambda x: x[2], reverse=True)[:5]
    # "Worst" = weakest everyday player: the lowest OVR inside each club's top 18.
    regulars: List[Tuple[Any, Any, float]] = []
    by_team: Dict[int, List[Tuple[Any, Any, float]]] = {}
    for row in all_players:
        by_team.setdefault(id(row[1]), []).append(row)
    for rows in by_team.values():
        regulars.extend(sorted(rows, key=lambda x: x[2], reverse=True)[:18])
    worst_players = sorted([r for r in regulars if r[2] > 0], key=lambda x: x[2])[:3]

    top_prospects = sorted(prospects, key=lambda x: (x[2], x[3]), reverse=True)[:5]
    oldest = max(all_players, key=lambda x: _age(x[0]), default=None)
    youngest = min([r for r in all_players if _age(r[0]) > 0], key=lambda x: _age(x[0]), default=None)

    cap_leaders: List[Dict[str, Any]] = []
    if with_cap:
        capped = sorted(((p, t, _cap(p)) for p, t, _ in all_players), key=lambda x: x[2], reverse=True)
        for p, t, c in capped[:5]:
            if c > 0:
                row = _player_row(p, t, _ovr(p))
                row["cap"] = round(c, 2)
                cap_leaders.append(row)

    # Divisions
    divisions: List[Dict[str, Any]] = []
    div_names: List[str] = []
    for r in ranked:
        if r["division"] and r["division"] not in div_names:
            div_names.append(r["division"])
    for d in div_names:
        clubs = [r for r in ranked if r["division"] == d]
        if not clubs:
            continue
        avg = round(sum(c["ovr"] for c in clubs) / len(clubs), 1)
        divisions.append({
            "name": d,
            "conference": clubs[0]["conference"],
            "avg": avg,
            "favorite": {"abbr": clubs[0]["abbr"], "name": clubs[0]["name"], "ovr": clubs[0]["ovr"]},
            "dark_horse": ({"abbr": clubs[len(clubs) // 2]["abbr"], "name": clubs[len(clubs) // 2]["name"],
                            "ovr": clubs[len(clubs) // 2]["ovr"]} if len(clubs) > 2 else None),
            "basement": {"abbr": clubs[-1]["abbr"], "name": clubs[-1]["name"], "ovr": clubs[-1]["ovr"]},
            "gap": round(clubs[0]["ovr"] - clubs[-1]["ovr"], 1),
        })
    divisions.sort(key=lambda d: d["avg"], reverse=True)

    # Your club
    mine = next((r for r in ranked if r["id"] == str(user_tid)), None)
    user_team_out: Optional[Dict[str, Any]] = None
    rivals: List[Dict[str, Any]] = []
    if mine:
        rank = mine.get("rank") or n
        if rank <= max(1, n // 6):
            tier, verdict = "contender", "Cup or bust. The window is wide open."
        elif rank <= n // 2:
            tier, verdict = "playoff", "A playoff team on paper. Every point matters."
        elif rank <= (3 * n) // 4:
            tier, verdict = "bubble", "Bubble club. Smart moves could tip it."
        else:
            tier, verdict = "rebuild", "Long season ahead. Time to build through the draft."
        my_pros = sorted([x for x in prospects if x[1] is not None and _tid(x[1]) == mine["id"]],
                         key=lambda x: x[2], reverse=True)
        div_rank = 1 + sum(1 for r in ranked if r["division"] == mine["division"] and r["ovr"] > mine["ovr"])
        user_team_out = {
            **{k: mine[k] for k in ("abbr", "name", "division", "conference", "ovr", "star", "goalie", "avg_age")},
            "rank": rank,
            "of": n,
            "division_rank": div_rank,
            "tier": tier,
            "verdict": verdict,
            "top_prospect": (
                {**_player_row(my_pros[0][0], my_pros[0][1], my_pros[0][3]), "pot": int(round(my_pros[0][2]))}
                if my_pros else None
            ),
        }
        same_div = [r for r in ranked if r["division"] == mine["division"] and r["id"] != mine["id"]]
        same_div.sort(key=lambda r: abs(r["ovr"] - mine["ovr"]))
        for r in same_div[:3]:
            delta = round(r["ovr"] - mine["ovr"], 1)
            rivals.append({
                "abbr": r["abbr"],
                "name": r["name"],
                "ovr": r["ovr"],
                "delta": delta,
                "star": (r.get("star") or {}).get("name"),
            })

    # Storylines (computed, not random — each line names its evidence)
    storylines: List[str] = []
    if ranked:
        top = ranked[0]
        storylines.append(f"{top['name']} open as the team to beat ({top['ovr']:.1f} OVR).")
        if n > 1 and ranked[0]["ovr"] - ranked[1]["ovr"] < 0.6:
            storylines.append(f"Dead heat at the top: {ranked[0]['abbr']} and {ranked[1]['abbr']} are separated by a hair.")
    if divisions:
        storylines.append(f"The {divisions[0]['name']} is the group of death ({divisions[0]['avg']:.1f} avg OVR).")
        weak = divisions[-1]
        storylines.append(f"The {weak['name']} looks wide open. {weak['favorite']['abbr']} lead a soft field.")
    if top_prospects:
        p, t, pot, o = top_prospects[0]
        storylines.append(f"Scouts are buzzing about {_name(p)} ({_team_abbr(t)}), age {_age(p)}, {pot:.0f} potential.")
    if oldest:
        storylines.append(f"{_name(oldest[0])} ({_team_abbr(oldest[1])}) is the league's elder statesman at {_age(oldest[0])}.")
    if youngest:
        storylines.append(f"{_name(youngest[0])} ({_team_abbr(youngest[1])}) is the youngest NHLer at {_age(youngest[0])}.")
    best_g = max((r for r in ranked if r.get("goalie")), key=lambda r: r["goalie"]["ovr"], default=None)
    if best_g:
        storylines.append(f"{best_g['goalie']['name']} ({best_g['abbr']}) is the best goalie on paper.")
    if cap_leaders:
        c = cap_leaders[0]
        storylines.append(f"{c['name']} cashes the biggest cheque: ${c['cap']:.2f}M a season.")
    if ranked:
        old = max(ranked, key=lambda r: r["avg_age"])
        young = min(ranked, key=lambda r: r["avg_age"] or 99)
        storylines.append(f"Oldest room: {old['abbr']} ({old['avg_age']}). Youngest: {young['abbr']} ({young['avg_age']}).")

    def _team_brief(r: Dict[str, Any]) -> Dict[str, Any]:
        return {"abbr": r["abbr"], "name": r["name"], "ovr": r["ovr"], "rank": r.get("rank"),
                "star": (r.get("star") or {}).get("name")}

    season_label = ""
    try:
        sy = int(season_y)
        season_label = f"{sy}-{str(sy + 1)[-2:]}"
    except (TypeError, ValueError):
        pass

    return {
        "created_iso": start_iso or "",
        "created_label": _pretty_date(start_iso) if start_iso else "",
        "season_label": season_label,
        "team_count": n,
        "player_count": len(all_players),
        "best_teams": [_team_brief(r) for r in ranked[:3]],
        "worst_teams": [_team_brief(r) for r in ranked[-3:][::-1]] if n > 3 else [],
        "best_players": [_player_row(p, t, o) for p, t, o in best_players],
        "worst_players": [_player_row(p, t, o) for p, t, o in worst_players],
        "top_prospects": [
            {**_player_row(p, t, o), "pot": int(round(pot))} for p, t, pot, o in top_prospects
        ],
        "cap_leaders": cap_leaders,
        "storylines": storylines[:8],
        "divisions": divisions,
        "rivals": rivals,
        "user_team": user_team_out,
    }


def _session_start_iso(session: Any) -> str:
    try:
        return str(franchise_sim._calendar_iso_for_day(session, 0) or "")
    except Exception:
        return ""


def _report_from_session(session: Any) -> Dict[str, Any]:
    cached = getattr(session, "_creation_report_cache", None)
    if isinstance(cached, dict):
        return cached
    league = getattr(getattr(session, "sim", None), "league", None)
    teams = list(getattr(league, "teams", None) or [])
    rep = _build_report(
        teams,
        str(getattr(session, "user_team_id", "") or ""),
        season_y=getattr(session, "season_calendar_year", None),
        start_iso=_session_start_iso(session),
        with_cap=True,
    )
    try:
        setattr(session, "_creation_report_cache", rep)
    except Exception:
        pass
    return rep


def _live_payload() -> Dict[str, Any]:
    with _LOCK:
        live = dict(_LIVE)
        steps = [dict(s) for s in (live.get("steps") or [])]
    out: Dict[str, Any] = {
        "ok": True,
        "source": "live" if live else "none",
        "ready": False,
        "steps": [{"key": s["key"], "line": s["line"]} for s in steps],
        "elapsed": round(time.time() - float(live.get("started_at") or time.time()), 1),
    }
    if not live.get("rosters_ready") or live.get("league") is None:
        return out
    keys = {s["key"] for s in steps}
    contracts_ready = "contracts" in keys
    cache_key = ("contracts" if contracts_ready else "rosters", id(live.get("league")))
    rep = _LIVE_CACHE.get(cache_key)
    if rep is None:
        try:
            start_iso = ""
            session = live.get("session")
            if session is not None:
                start_iso = _session_start_iso(session)
            if not start_iso:
                cal = live.get("nhl_cal") or []
                if cal and isinstance(cal[0], dict):
                    start_iso = str(cal[0].get("iso") or cal[0].get("date") or "")
            user_team = live.get("user_team")
            rep = _build_report(
                list(getattr(live.get("league"), "teams", None) or []),
                _tid(user_team) if user_team is not None else str(live.get("uid") or ""),
                season_y=live.get("season_y"),
                start_iso=start_iso,
                with_cap=contracts_ready,
            )
            # Re-roll once the calendar arrives so the creation date is filled in.
            if start_iso:
                _LIVE_CACHE[cache_key] = rep
        except Exception:
            log.debug("live creation report failed", exc_info=True)
            return out
    out["ready"] = True
    out["report"] = rep
    return out


def get_creation_report(session_id: Optional[str]) -> Dict[str, Any]:
    sid = str(session_id or "").strip()
    if sid:
        session = get_session(sid)
        if session is not None:
            try:
                rep = _report_from_session(session)
                return {"ok": True, "source": "session", "ready": True, "steps": [], "report": rep}
            except Exception:
                log.debug("session creation report failed", exc_info=True)
    try:
        return _live_payload()
    except Exception:
        log.debug("creation report live payload failed", exc_info=True)
        return {"ok": True, "source": "none", "ready": False, "steps": []}


if router is not None:

    @router.get("/api/franchise/creation-report")
    def creation_report_endpoint(
        x_franchise_session: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        return get_creation_report(x_franchise_session)
