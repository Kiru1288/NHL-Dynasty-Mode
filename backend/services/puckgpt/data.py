"""What PuckGPT reads: the live save, plus one JSON archive built on demand.

Nothing here runs during the sim. The live view is assembled when a question
is asked and cached until the roster or stats revision moves. The archive file
(``data/puckgpt/<session>.json``) is written only from the PuckGPT text box,
holds finished seasons, and gains a season when a new one has been finalised.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .nlu import Person, World

log = logging.getLogger(__name__)

_ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "puckgpt")
_ARCHIVE_VERSION = 1
_MEM: Dict[str, Dict[str, Any]] = {}   # session id -> {"key": ..., "view": LiveView}
_ARCH_MEM: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- player fields
def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def p_name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", None) or "Player")


def p_pos(p: Any) -> str:
    ident = getattr(p, "identity", None)
    raw = getattr(ident, "position", None) if ident is not None else getattr(p, "position", "")
    raw = getattr(raw, "value", raw)
    return str(raw or "").upper()


def p_shoots(p: Any) -> str:
    ident = getattr(p, "identity", None)
    raw = getattr(ident, "shoots", None) if ident is not None else None
    return str(getattr(raw, "value", raw) or "").upper()[:1]


def p_age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    return int(_num(getattr(ident, "age", None) if ident is not None else getattr(p, "age", 0)))


def p_ovr(p: Any) -> float:
    try:
        from app.sim_engine.franchise.storyline_conduct import get_base_ovr_display

        return float(get_base_ovr_display(p))
    except Exception:
        v = _num(getattr(p, "overall", 0))
        return v * 99.0 if 0 < v <= 1.5 else v


def p_pot(p: Any) -> float:
    ratings = getattr(p, "ratings", None)
    raw = ratings.get("dev_potential") if isinstance(ratings, dict) else None
    if raw in (None, 0, 0.0):
        raw = getattr(p, "potential", None)
    v = _num(raw)
    return v * 99.0 if 0 < v <= 1.5 else v


def p_contract(p: Any) -> Dict[str, Any]:
    c = getattr(p, "contract", None)
    if not isinstance(c, dict):
        c = {}
    cap = _num(c.get("cap_hit_m", c.get("aav_m", getattr(p, "cap_hit_m", 0))))
    return {
        "cap_hit_m": round(cap, 3),
        "years_left": int(_num(c.get("years_remaining", c.get("years", 0)))),
        "expiry_year": int(_num(c.get("expiry_year", 0))) or None,
        "status": str(c.get("expiry_status") or c.get("rights_status") or getattr(p, "rights_status", "") or "").upper(),
        "ntc": bool(c.get("ntc")), "nmc": bool(c.get("nmc")),
        "type": str(c.get("contract_type") or c.get("type") or ""),
    }


def p_role(p: Any) -> str:
    for attr in ("playstyle", "archetype"):
        v = getattr(p, attr, None)
        v = getattr(v, "value", v)
        if v:
            return str(v)
    return ""


# --------------------------------------------------------------------------- live view
class LiveView:
    """One pass over the save, cached per roster/stats revision."""

    def __init__(self, session: Any) -> None:
        self.session = session
        sim = getattr(session, "sim", None)
        self.league = getattr(sim, "league", None)
        self.user_tid = str(getattr(session, "user_team_id", "") or "")
        self.season = int(getattr(session, "season_calendar_year", 0) or 0)
        self.teams: Dict[str, Any] = {str(k): v for k, v in (getattr(session, "team_by_id", None) or {}).items()}
        self.team_meta: Dict[str, Dict[str, str]] = {}
        for tid, t in self.teams.items():
            self.team_meta[tid] = {
                "abbr": str(getattr(t, "abbreviation", None) or getattr(t, "abbr", None) or tid),
                "name": str(getattr(t, "full_name", None) or f"{getattr(t, 'city', '')} {getattr(t, 'name', '')}".strip()),
                "city": str(getattr(t, "city", "") or ""), "nickname": str(getattr(t, "name", "") or ""),
            }
        self.players: Dict[str, Any] = {}
        self.where: Dict[str, Tuple[str, str]] = {}  # pid -> (team id or "", level)
        for tid, t in self.teams.items():
            for level, attr in (("NHL", "roster"), ("AHL", "ahl_roster"), ("ECHL", "echl_roster"), ("Prospect", "prospect_pool")):
                for p in list(getattr(t, attr, None) or []):
                    pid = str(getattr(p, "id", "") or "")
                    if pid and not getattr(p, "retired", False) and pid not in self.players:
                        self.players[pid] = p
                        self.where[pid] = (tid, level)
        for attr in ("free_agents", "overseas_free_agents"):
            for p in list(getattr(self.league, attr, None) or []):
                pid = str(getattr(p, "id", "") or "")
                if pid and pid not in self.players and not getattr(p, "retired", False):
                    self.players[pid] = p
                    self.where[pid] = ("", "FA")
        ledger = getattr(session, "player_season_stats", None)
        if not isinstance(ledger, dict) or not ledger:
            ledger = getattr(self.league, "player_season_stats", None)
        self.stats: Dict[str, Dict[str, Any]] = ledger if isinstance(ledger, dict) else {}
        self._rows: Dict[str, Dict[str, Any]] = {}
        self.world = self._world()

    def _world(self) -> World:
        people = []
        for pid, p in self.players.items():
            num = getattr(p, "sweater_number", None)
            people.append(Person(pid, p_name(p), self.where[pid][0], p_pos(p), int(num) if str(num or "").isdigit() else None, "", ""))
        return World(people=people, teams=self.team_meta, user_team_id=self.user_tid).index()

    def row(self, pid: str) -> Dict[str, Any]:
        r = self._rows.get(pid)
        if r is not None:
            return r
        p = self.players.get(pid)
        if p is None:
            return {}
        tid, level = self.where.get(pid, ("", ""))
        st = self.stats.get(pid) or {}
        gp = int(_num(st.get("gp")))
        r = {
            "id": pid, "name": p_name(p), "pos": p_pos(p), "shoots": p_shoots(p), "age": p_age(p),
            "ovr": round(p_ovr(p), 1), "pot": round(p_pot(p), 1), "team_id": tid,
            "team": self.team_meta.get(tid, {}).get("abbr", "FA" if level == "FA" else ""), "level": level,
            "role": p_role(p), "contract": p_contract(p), "gp": gp,
            "g": int(_num(st.get("g"))), "a": int(_num(st.get("a"))), "pts": int(_num(st.get("pts"))),
            "plus_minus": int(_num(st.get("plus_minus"))), "pim": int(_num(st.get("pim"))),
            "sog": int(_num(st.get("sog"))), "hit": int(_num(st.get("hit"))), "blk": int(_num(st.get("blk"))),
            "toi_pg": round(_num(st.get("toi_sec")) / gp / 60.0, 1) if gp else 0.0,
            "w": int(_num(st.get("w"))), "so": int(_num(st.get("so"))),
            "ppg": int(_num(st.get("ppg"))), "ppp": int(_num(st.get("ppg")) + _num(st.get("ppa"))),
            "shg": int(_num(st.get("shg"))), "gwg": int(_num(st.get("gwg"))),
            "fo_total": int(_num(st.get("fow")) + _num(st.get("fol"))),
            "fo_pct": round(_num(st.get("fow")) / max(1.0, _num(st.get("fow")) + _num(st.get("fol"))), 3)
            if (_num(st.get("fow")) + _num(st.get("fol"))) > 0 else None,
            "sv_pct": round(1.0 - _num(st.get("goalie_ga", st.get("ga"))) / _num(st.get("shots_against") or st.get("goalie_shots_against"), 1.0), 3)
            if p_pos(p) == "G" and _num(st.get("shots_against") or st.get("goalie_shots_against")) > 0 else None,
            "season_start_ovr": _num(getattr(p, "season_start_ovr", None), 0.0) or None,
            "injured": _is_injured(p),
        }
        self._rows[pid] = r
        return r

    def game_log(self, pid: str) -> Dict[str, Any]:
        """This season's per-game lines from the box-score book (faceoffs and recent form live here)."""
        cache = self.__dict__.setdefault("_game_logs", {})
        if pid not in cache:
            try:
                from app.sim_engine.franchise.serialization import get_player_game_log_payload

                cache[pid] = get_player_game_log_payload(self.session, pid) or {}
            except Exception:
                log.debug("game log read failed", exc_info=True)
                cache[pid] = {}
        return cache[pid]

    def rows(self, ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        return [self.row(pid) for pid in (ids if ids is not None else self.players.keys()) if pid in self.players]

    def team_rows(self, tid: str, levels: Tuple[str, ...] = ("NHL",)) -> List[Dict[str, Any]]:
        return [self.row(pid) for pid, (t, lv) in self.where.items() if t == tid and lv in levels]

    def standings(self) -> List[Dict[str, Any]]:
        try:
            from services.franchise_sim import _build_standings_rows

            return list(_build_standings_rows(self.session) or [])
        except Exception:
            log.debug("standings read failed", exc_info=True)
            return []

    def team_games(self, tid: str) -> List[Tuple[str, bool, int, int]]:
        """This season's finals for one club: (date, won, gf, ga)."""
        out = []
        for box in list(getattr(self.session, "game_results", None) or []):
            if not isinstance(box, dict):
                continue
            h, a = str(box.get("home_id") or ""), str(box.get("away_id") or "")
            if tid not in (h, a):
                continue
            hg, ag = int(_num(box.get("home_goals", box.get("home_score")))), int(_num(box.get("away_goals", box.get("away_score"))))
            gf, ga = (hg, ag) if tid == h else (ag, hg)
            out.append((str(box.get("calendar_iso") or box.get("iso") or "")[:10], gf > ga, gf, ga))
        out.sort()
        return out


def _is_injured(p: Any) -> bool:
    try:
        from services.franchise_sim import _is_player_live_injured

        return bool(_is_player_live_injured(p))
    except Exception:
        return False


def live_view(session: Any) -> LiveView:
    sid = str(getattr(session, "session_id", "") or id(session))
    key = (
        int(getattr(session, "_roster_revision", 0) or 0), int(getattr(session, "_stats_revision", 0) or 0),
        int(getattr(session, "calendar_days_finished", 0) or 0), int(getattr(session, "season_calendar_year", 0) or 0),
        str(getattr(session, "phase", "")),
    )
    with _LOCK:
        hit = _MEM.get(sid)
        if hit and hit["key"] == key and hit["view"].session is session:
            return hit["view"]
    # A sim running in another request can resize rosters mid-read; retry a couple of times.
    for attempt in range(3):
        try:
            view = LiveView(session)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(0.05)
    with _LOCK:
        _MEM[sid] = {"key": key, "view": view}
    return view


# --------------------------------------------------------------------------- archive file
def _archive_path(session: Any) -> str:
    sid = "".join(ch for ch in str(getattr(session, "session_id", "") or "save") if ch.isalnum() or ch in "-_")
    return os.path.join(_ARCHIVE_DIR, f"{sid or 'save'}.json")


def _season_section(session: Any, lg: Dict[str, Any], year: int) -> Dict[str, Any]:
    s = (lg.get("seasons") or {}).get(str(year)) or {}
    lines = []
    for pid, pr in (lg.get("players") or {}).items():
        for ln in pr.get("seasons") or []:
            if ln.get("season") == year and ln.get("source") == "simulated":
                lines.append({"id": pid, "name": pr.get("name"), "pos": pr.get("position"), **{
                    k: ln.get(k) for k in ("team_id", "scope", "gp", "g", "a", "pts", "plus_minus", "pim", "w", "so", "sv_pct") if ln.get(k) is not None}})
    trades = [t for t in (lg.get("trades") or {}).values() if int(t.get("season") or 0) == year]
    return {
        "season": year, "label": s.get("label"), "champion_id": s.get("champion_id"), "finalist_id": s.get("finalist_id"),
        "standings": s.get("standings") or {}, "series": s.get("series") or s.get("playoff_series") or [],
        "awards": s.get("awards") or [], "draft": s.get("draft") or [], "player_lines": lines, "trades": trades,
    }


def archive(session: Any) -> Dict[str, Any]:
    """Load the save's archive JSON, adding any finalised season it does not hold yet.

    Called only from PuckGPT. First call per server session reads the file; later
    calls are a dict lookup unless a new season has been finalised since.
    """
    sid = str(getattr(session, "session_id", "") or id(session))
    try:
        from services.legacy_hall import ensure_legacy

        lg = ensure_legacy(session)
    except Exception:
        lg = getattr(session, "legacy", None) or {}
    # A season row fills in by stage (standings, cup, awards, draft); its signature
    # changes as stages land, and only changed seasons are rebuilt.
    done = sorted(
        [int(k), str(s.get("champion_id") or ""), len(s.get("awards") or []), len(s.get("draft") or []), bool(s.get("complete"))]
        for k, s in (lg.get("seasons") or {}).items() if s.get("standings")
    )
    with _LOCK:
        cached = _ARCH_MEM.get(sid)
    if cached is not None and cached.get("meta", {}).get("seasons") == done:
        return cached
    path = _archive_path(session)
    doc: Dict[str, Any] = {}
    if cached is None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            doc = {}
    else:
        doc = cached
    if doc.get("meta", {}).get("version") != _ARCHIVE_VERSION:
        doc = {"meta": {"version": _ARCHIVE_VERSION, "seasons": []}, "seasons": {}}
    have = {tuple(sig) for sig in doc["meta"].get("seasons") or [] if isinstance(sig, list)}
    missing = [sig for sig in done if tuple(sig) not in have]
    for sig in missing:
        doc["seasons"][str(sig[0])] = _season_section(session, lg, sig[0])
    doc["meta"]["seasons"] = done
    doc["meta"]["user_team"] = str(getattr(session, "user_team_id", "") or "")
    if missing:
        try:
            os.makedirs(_ARCHIVE_DIR, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, separators=(",", ":"), default=str)
            os.replace(tmp, path)
        except Exception:
            log.debug("puckgpt archive write failed", exc_info=True)
    with _LOCK:
        _ARCH_MEM[sid] = doc
    return doc
