"""League social feed — Puckr (X/Twitter-style), IceHole (Reddit-style) and burner accounts.

Every post is generated from real sim state: the day's box scores, per-player game lines
(diffed from ``session.player_season_stats``), standings / playoff odds, injuries, trades,
signings, trade demands, storylines, governance votes, draft, awards and the user's moves.

Storage
-------
* ``session.social_posts``    — Puckr + burner posts (``v == 2`` rows), newest last.
* ``session.reddit_threads``  — IceHole threads with comment trees (``v == 2`` rows).
* ``session.social_feed_state`` — small cursor/snapshot dict (stat snapshot, seen ids,
  team tallies). Archive is pruned to ``ARCHIVE_DAYS`` and hard caps.

Entry points
------------
* ``run_social_feed_day(session, day_idx, day_meta)`` — once per simulated calendar day.
* ``ensure_social_feed_current(session)`` — cheap catch-up for event sources (trades,
  signings, playoffs, offseason stages, governance...) used by the feed endpoint.
* ``build_social_feed_response(...)`` / ``get_social_thread(...)`` — paginated API payloads.
* ``record_gm_burner_post(session, result)`` — the GM's own burner posts land in the feed.

Cost per day is O(players with stats + games + small tails of event logs); there are no
player-pair loops.
"""

from __future__ import annotations

import logging
import math
import random
import re
import threading
import zlib
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

_log = logging.getLogger(__name__)

FEED_VERSION = 2
ARCHIVE_DAYS = 60
MAX_POSTS = 2000
MAX_THREADS = 400
MAX_SEEN = 500
STATE_ATTR = "social_feed_state"

_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# Static flavour tables (fictional accounts — no real journalists are impersonated)
# ---------------------------------------------------------------------------

TEAM_COLORS: Dict[str, str] = {
    "ANA": "#f47a20", "BOS": "#ffb81c", "BUF": "#003087", "CAR": "#cc0000", "CBJ": "#002654",
    "CGY": "#c8102e", "CHI": "#cf0a2c", "COL": "#6f263d", "DAL": "#006847", "DET": "#ce1126",
    "EDM": "#ff4c00", "FLA": "#c8102e", "LAK": "#a2aaad", "MIN": "#154734", "MTL": "#af1e2d",
    "NJD": "#ce1126", "NSH": "#ffb81c", "NYI": "#00539b", "NYR": "#0038a8", "OTT": "#c52032",
    "PHI": "#f74902", "PIT": "#fcb514", "SEA": "#99d9d9", "SJS": "#006d75", "STL": "#002f87",
    "TBL": "#002868", "TOR": "#00205b", "UTA": "#6cace4", "VAN": "#00843d", "VGK": "#b4975a",
    "WPG": "#041e42", "WSH": "#c8102e",
}

FAN_NICK: Dict[str, str] = {
    "OTT": "Sens", "MTL": "Habs", "TOR": "Leafs", "TBL": "Bolts", "PIT": "Pens", "WSH": "Caps",
    "NYI": "Isles", "CAR": "Canes", "COL": "Avs", "NSH": "Preds", "UTA": "Mammoth", "CBJ": "Jackets",
    "CHI": "Hawks", "DET": "Wings", "LAK": "Kings", "ANA": "Ducks", "SJS": "Sharks", "VGK": "Knights",
    "SEA": "Kraken", "EDM": "Oilers", "CGY": "Flames", "VAN": "Nucks", "WPG": "Jets", "MIN": "Wild",
    "STL": "Blues", "DAL": "Stars", "BOS": "Bruins", "BUF": "Sabres", "NJD": "Devils", "NYR": "Rangers",
    "PHI": "Flyers", "FLA": "Cats",
}

BIG_MARKET_BUMP: Dict[str, float] = {"TOR": 0.3, "MTL": 0.3, "NYR": 0.15, "BOS": 0.12, "CHI": 0.1, "EDM": 0.12, "VAN": 0.1, "DET": 0.08, "PHI": 0.08}

# National voices reuse the game's existing newsroom (storyline_engine.MEDIA_REPORTERS).
INSIDERS: Dict[str, Dict[str, Any]] = {
    "ellison": {"name": "Mark Ellison", "handle": "@MarkEllisonNHL", "outlet": "NorthStar Hockey", "badge": "insider", "beat": "trades"},
    "reid": {"name": "Mason Reid", "handle": "@CapReid", "outlet": "PuckFinance", "badge": "insider", "beat": "cap"},
    "petrov": {"name": "Alex Petrov", "handle": "@PetrovProspects", "outlet": "Future Ice", "badge": "insider", "beat": "draft"},
    "lee": {"name": "Jenna Lee", "handle": "@JennaLeeReports", "outlet": "National Sports Desk", "badge": "insider", "beat": "news"},
    "howe": {"name": "Sam Howe", "handle": "@CreaseReportSam", "outlet": "Crease Report", "badge": "insider", "beat": "goalies"},
    "knox": {"name": "Derek Knox", "handle": "@KnoxOnHockey", "outlet": "NBN", "badge": "insider", "beat": "analysis"},
    "hart": {"name": "Chris Hart", "handle": "@HotTakeHart", "outlet": "Hot Take TV", "badge": "media", "beat": "takes"},
    "vargas": {"name": "Tessa Vargas", "handle": "@TessaVargasHKY", "outlet": "Slot Line Media", "badge": "insider", "beat": "trades"},
}

OFFICIAL = {"id": "leaguewire", "name": "League Wire", "handle": "@LeagueWire", "type": "official", "badge": "official", "outlet": "League Office"}

STATS_ACCOUNTS: List[Dict[str, str]] = [
    {"id": "xgdaily", "name": "Expected Goals Daily", "handle": "@xGoalsDaily"},
    {"id": "shotshare", "name": "Shot Share Lab", "handle": "@ShotShareLab"},
    {"id": "pdowatch", "name": "PDO Regression Watch", "handle": "@PDOwatch"},
    {"id": "netfront", "name": "Net Front Numbers", "handle": "@NetFrontNumbers"},
]

MEME_ACCOUNTS: List[Dict[str, str]] = [
    {"id": "crossbar", "name": "Crossbar Daily", "handle": "@CrossbarDaily"},
    {"id": "brainrot", "name": "Hockey Brain Rot", "handle": "@HockeyBrainRot"},
    {"id": "gitruthers", "name": "Goalie Interference Truthers", "handle": "@GITruthers"},
    {"id": "bardown", "name": "Bardown Bandit", "handle": "@BardownBandit"},
]

BURNER_ACCOUNTS: List[Dict[str, str]] = [
    {"id": "notascout", "name": "not a scout", "handle": "@notascout_44", "voice": "scout"},
    {"id": "pressbox", "name": "pressbox ghost", "handle": "@pressbox_ghost", "voice": "media"},
    {"id": "capleaks", "name": "capsheet leaks", "handle": "@capsheetleaks", "voice": "cap"},
    {"id": "zamboni", "name": "zamboni insider", "handle": "@zamboni_insider", "voice": "rink"},
    {"id": "ahlbus", "name": "ahl bus driver", "handle": "@ahlbusdriver", "voice": "minors"},
]

BEAT_FIRST = ["Mike", "Sarah", "Ryan", "Elise", "Dan", "Kara", "Pat", "Jules", "Ben", "Nadia", "Chris", "Leah", "Tom", "Maya", "Grant", "Erin", "Josh", "Amelie", "Luke", "Hana", "Matt", "Dev", "Nora", "Will"]
BEAT_LAST = ["Garrioch", "Halvorsen", "Bellamy", "Kowal", "Dufresne", "Ashby", "McAdam", "Lindgren", "Pruitt", "Ostrowski", "Calder", "Mercer", "Ferland", "Quigley", "Strand", "Toews-Hall", "Varga", "Whitlock", "Beaudry", "Kerr", "Nakamura", "Sutherland", "Okoro", "Delisle"]
BEAT_OUTLET = ["Gazette", "Ledger", "Post", "Herald", "Sun", "Courier", "Times", "Daily", "Press", "Sports Desk"]

FAN_PERSONAS: List[Dict[str, Any]] = [
    {"key": "homer", "suffixes": ["Army", "Nation", "Faithful"], "bias": 0.35},
    {"key": "doomer", "suffixes": ["Doomer", "Pessimist", "Suffering"], "bias": -0.35},
    {"key": "ironic", "suffixes": ["Brainrot", "Hot Takes", "Burner Club"], "bias": 0.0},
]

REDDIT_PREFIX = ["Bardown", "Crossbar", "FiveHole", "Grinder", "FourthLine", "Slot", "Rink", "Boards", "Blueline", "Tape2Tape", "Dangle", "Sauce", "Celly", "Biscuit", "Twig", "Barn", "Zamboni", "Pylon", "Toe Drag", "Wraparound"]
REDDIT_SUFFIX = ["Rat", "Enjoyer", "Truther", "Merchant", "Guy", "Andy", "Lad", "Sicko", "Goblin", "Dad", "Bandit", "Gremlin", "Lifer", "Prophet", "Haver", "Fan", "Hater", "Watcher"]

OFFSEASON_STAGE_DATES: Dict[str, Tuple[int, int]] = {
    "awards": (6, 22), "retirements": (6, 23), "board_of_governors": (6, 24), "salary_cap": (6, 25),
    "development_report": (6, 25), "draft_lottery": (6, 26), "draft_combine": (6, 26), "draft": (6, 27),
    "draft_review": (6, 28), "prospect_rights": (6, 29), "re_sign": (6, 30), "free_agency": (7, 1),
    "next_season_reveal": (8, 20),
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _h(*parts: Any) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8", "ignore")) & 0xFFFFFFFF


def _sf(v: Any, d: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return d
        return x
    except (TypeError, ValueError):
        return d


def _si(v: Any, d: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _iso_shift(iso: str, days: int) -> str:
    try:
        return (date.fromisoformat(str(iso)[:10]) + timedelta(days=int(days))).isoformat()
    except Exception:
        return str(iso or "")[:10]


def _iso_days_between(a: str, b: str) -> int:
    try:
        return (date.fromisoformat(str(b)[:10]) - date.fromisoformat(str(a)[:10])).days
    except Exception:
        return 0


def _pretty_date(iso: str) -> str:
    try:
        d = date.fromisoformat(str(iso)[:10])
        return d.strftime("%b %d, %Y").replace(" 0", " ")
    except Exception:
        return str(iso or "")


def _short_date(iso: str) -> str:
    try:
        d = date.fromisoformat(str(iso)[:10])
        return d.strftime("%b %d").replace(" 0", " ")
    except Exception:
        return str(iso or "")


def _money(m: float) -> str:
    m = _sf(m)
    if m >= 10:
        return f"${m:.1f}M"
    if m >= 1:
        return f"${m:.2f}M".replace(".00M", "M")
    return f"${int(round(m * 1000))}K"


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _svpct(x: float) -> str:
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0") else s


def _ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _plural(n: int, word: str, plural: Optional[str] = None) -> str:
    return f"{n} {word if int(n) == 1 else (plural or word + 's')}"


def _last(name: str) -> str:
    parts = str(name or "").split()
    return parts[-1] if parts else str(name or "")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(text or ""))


def _hhmm(minutes: int) -> str:
    minutes = int(_clamp(minutes, 0, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _pick(rng: random.Random, seq: List[Any]) -> Any:
    return seq[rng.randrange(len(seq))] if seq else None


_FMT_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _fill(template: str, ctx: Dict[str, Any]) -> Optional[str]:
    """Fill ``{key}`` slots; return None when a slot is missing/empty (template is skipped)."""
    missing = False

    def repl(m: "re.Match[str]") -> str:
        nonlocal missing
        v = ctx.get(m.group(1))
        if v is None or v == "":
            missing = True
            return ""
        return str(v)

    out = _FMT_RE.sub(repl, template)
    if missing:
        return None
    return re.sub(r"\s+([.,!?;:])", r"\1", re.sub(r"[ \t]+", " ", out)).strip()


_FP_RE = re.compile(r"[A-Z][\w'.-]*|\d+|[^\w\s]")


def _fingerprint(text: str) -> str:
    """Shape of a line with names and numbers stripped, so the same template filled
    for two fanbases counts as a repeat."""
    return " ".join(_FP_RE.sub("", str(text or "")).lower().split())[:40]


def _recent_fps(st: Dict[str, Any]) -> List[str]:
    return st.setdefault("recent_fp", [])


def _remember_fp(st: Dict[str, Any], text: Optional[str]) -> None:
    if not text:
        return
    rows = _recent_fps(st)
    rows.append(_fingerprint(text))
    if len(rows) > 260:
        del rows[: len(rows) - 260]


def _pick_fresh(st: Dict[str, Any], rng: random.Random, options: List[str]) -> str:
    """Pick an option whose shape hasn't been used recently (falls back to any)."""
    opts = [o for o in options if o]
    if not opts:
        return ""
    recent = set(_recent_fps(st)[-160:])
    fresh = [o for o in opts if _fingerprint(o) not in recent]
    choice = rng.choice(fresh or opts)
    _remember_fp(st, choice)
    return choice


def _compose_fresh(st: Dict[str, Any], rng: random.Random, templates: List[str], ctx: Dict[str, Any], limit: int = 280) -> Optional[str]:
    filled = [t for t in (_fill(tp, ctx) for tp in templates) if t and len(t) <= limit]
    if not filled:
        return None
    return _pick_fresh(st, rng, filled)


def _compose(rng: random.Random, templates: List[str], ctx: Dict[str, Any], limit: int = 280) -> Optional[str]:
    pool = list(templates)
    rng.shuffle(pool)
    for t in pool:
        txt = _fill(t, ctx)
        if txt:
            return txt[:limit]
    return None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _state(session: Any) -> Dict[str, Any]:
    st = getattr(session, STATE_ATTR, None)
    if not isinstance(st, dict) or int(st.get("version") or 0) != FEED_VERSION:
        st = {
            "version": FEED_VERSION,
            "salt": _h(getattr(session, "session_id", ""), "social-feed"),
            "seq": 0,
            "season": None,
            "last_day": -1,
            "last_ts": "",
            "snap": {},
            "teams": {},
            "nhl_ids": {},
            "seen": {},
            "flags": {},
            "bootstrapped": False,
        }
        try:
            setattr(session, STATE_ATTR, st)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        # Legacy (pre-v2) posts were template/filler rows — drop them on migration.
        try:
            session.social_posts = [p for p in list(getattr(session, "social_posts", None) or []) if isinstance(p, dict) and int(p.get("v") or 0) == FEED_VERSION]
            session.reddit_threads = [t for t in list(getattr(session, "reddit_threads", None) or []) if isinstance(t, dict) and int(t.get("v") or 0) == FEED_VERSION]
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return st


def _seen(st: Dict[str, Any], source: str) -> Tuple[List[str], set]:
    rows = st.setdefault("seen", {}).setdefault(source, [])
    return rows, set(rows)


def _mark_seen(st: Dict[str, Any], source: str, key: str) -> None:
    rows = st.setdefault("seen", {}).setdefault(source, [])
    rows.append(str(key))
    if len(rows) > MAX_SEEN:
        del rows[: len(rows) - MAX_SEEN]


def _flag(st: Dict[str, Any], key: str) -> bool:
    return bool(st.setdefault("flags", {}).get(key))


def _set_flag(st: Dict[str, Any], key: str) -> None:
    flags = st.setdefault("flags", {})
    flags[key] = 1
    if len(flags) > 400:
        for k in list(flags.keys())[:100]:
            flags.pop(k, None)


# ---------------------------------------------------------------------------
# League context
# ---------------------------------------------------------------------------


def _team_info(session: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    utid = str(getattr(session, "user_team_id", "") or "")
    for tid, tm in (getattr(session, "team_by_id", None) or {}).items():
        tid = str(tid)
        abbr = str(getattr(tm, "abbr", "") or getattr(tm, "abbreviation", "") or "").upper()
        city = str(getattr(tm, "city", "") or "")
        nick = str(getattr(tm, "name", "") or "")
        full = str(getattr(tm, "full_name", "") or f"{city} {nick}".strip())
        if city and nick and not full.endswith(nick):
            full = f"{city} {nick}"
        market = getattr(tm, "market", None)
        size = str(getattr(market, "market_size", "medium") or "medium")
        pressure = _sf(getattr(market, "media_pressure", 0.55), 0.55)
        mkt = {"small": 0.85, "medium": 1.0, "large": 1.28}.get(size, 1.0) * (1.0 + (pressure - 0.55) * 0.6)
        mkt += BIG_MARKET_BUMP.get(abbr, 0.0)
        coach = getattr(tm, "coach", None)
        coach_name = str(getattr(coach, "name", "") or "")
        if tid == utid:
            coach_name = str(getattr(session, "head_coach_name", "") or coach_name)
        if coach_name and len(coach_name) <= 3:
            coach_name = ""  # placeholder coach names ("AB") read badly in copy
        sub_slug = _slug(nick.split()[-1] if nick else abbr) or "hockey"
        out[tid] = {
            "id": tid,
            "abbr": abbr,
            "city": city or full,
            "nick": nick or abbr,
            "full": full or abbr,
            "fan": FAN_NICK.get(abbr) or (nick.split()[-1] if nick else abbr),
            "sub": f"r/{sub_slug}",
            "market": round(_clamp(mkt, 0.7, 1.7), 3),
            "coach": coach_name,
            "coach_security": _sf(getattr(coach, "job_security", 0.5), 0.5),
            "coach_hot": bool(getattr(coach, "hot_seat", False)),
            "gm": str(getattr(tm, "gm_name", "") or ""),
            "window": str(getattr(tm, "window", "") or getattr(tm, "gm_window", "") or ""),
            "conf": str(getattr(tm, "conference", "") or ""),
            "div": str(getattr(tm, "division", "") or ""),
            "color": TEAM_COLORS.get(abbr, "#5d7a8c"),
            "cap_space": _sf(getattr(tm, "cap_space_m", getattr(tm, "cap_space", 0.0)), 0.0),
            "is_user": tid == utid,
        }
    return out


def _standings(session: Any, teams: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    st = getattr(session, "standings", None)
    recs = getattr(st, "records", None) or {}
    season_len = max(10, _si(getattr(session, "games_per_team_schedule", 82), 82))
    rows: Dict[str, Dict[str, Any]] = {}
    items = recs.items() if isinstance(recs, dict) else []
    for tid, rr in items:
        tid = str(tid)
        if tid not in teams:
            continue
        w = _si(getattr(rr, "wins", 0))
        l = _si(getattr(rr, "losses", 0))
        o = _si(getattr(rr, "otl", 0))
        gp = w + l + o
        pts = _si(getattr(rr, "points", 2 * w + o))
        l10 = [str(x).upper()[:1] for x in (getattr(rr, "last_10", None) or [])]
        rows[tid] = {
            "w": w, "l": l, "o": o, "gp": gp, "pts": pts,
            "gf": _si(getattr(rr, "gf", 0)), "ga": _si(getattr(rr, "ga", 0)),
            "row": _si(getattr(rr, "row", w)),
            "rec": f"{w}-{l}-{o}",
            "l10": f"{l10.count('W')}-{l10.count('L')}-{len(l10) - l10.count('W') - l10.count('L')}" if l10 else "",
            "pct": (pts / (2.0 * gp)) if gp else 0.5,
            "conf": teams[tid]["conf"],
            "div": teams[tid]["div"],
        }
    if not rows:
        return rows
    order = sorted(rows.keys(), key=lambda t: (-rows[t]["pts"], -rows[t]["pct"], -rows[t]["row"], t))
    for i, tid in enumerate(order):
        rows[tid]["lg_rank"] = i + 1
    confs: Dict[str, List[str]] = {}
    for tid in order:
        confs.setdefault(rows[tid]["conf"], []).append(tid)
    for conf, tids in confs.items():
        for i, tid in enumerate(tids):
            r = rows[tid]
            r["conf_rank"] = i + 1
            remaining = max(0, season_len - r["gp"])
            r["proj"] = r["pts"] + r["pct"] * 2.0 * remaining
            r["max_pts"] = r["pts"] + 2 * remaining
        if len(tids) >= 9:
            eighth, ninth = rows[tids[7]], rows[tids[8]]
            cut_proj = (eighth["proj"] + ninth["proj"]) / 2.0
            for tid in tids:
                r = rows[tid]
                remaining = max(0, season_len - r["gp"])
                sd = max(1.2, 8.0 * math.sqrt(max(remaining, 1) / float(season_len)))
                z = (r["proj"] - cut_proj) / sd
                odds = 1.0 / (1.0 + math.exp(-1.7 * z))
                if r["conf_rank"] <= 8 and r["pts"] > ninth["max_pts"]:
                    odds = 1.0
                if r["conf_rank"] > 8 and r["max_pts"] < eighth["pts"]:
                    odds = 0.0
                r["odds"] = _clamp(odds, 0.0, 1.0)
                r["back"] = (eighth["pts"] - r["pts"]) if r["conf_rank"] > 8 else (r["pts"] - ninth["pts"])
                r["in"] = r["conf_rank"] <= 8
        else:
            for tid in tids:
                rows[tid].update({"odds": 0.5, "back": 0, "in": rows[tid]["conf_rank"] <= max(1, len(tids) // 2)})
    return rows


class _Pass:
    """Per-pass context: teams, standings, player lookups and the new-post buffers."""

    def __init__(self, session: Any, *, iso: str, day_idx: int, next_iso: str = "", mode: str = "day") -> None:
        self.session = session
        self.st = _state(session)
        self.iso = str(iso or "")[:10]
        self.next_iso = str(next_iso or self.iso)[:10]
        self.day_idx = int(day_idx)
        self.mode = mode
        self.utid = str(getattr(session, "user_team_id", "") or "")
        self.season = _si(getattr(session, "season_calendar_year", 0))
        self.phase = str(getattr(session, "phase", "") or "").lower()
        self.teams = _team_info(session)
        self.stand = _standings(session, self.teams)
        self.posts: List[Dict[str, Any]] = []
        self.threads: List[Dict[str, Any]] = []
        self.rng = random.Random(_h(self.st.get("salt"), "pass", self.iso, self.day_idx, mode))
        self._pidx: Optional[Dict[str, Any]] = None
        self.pss = getattr(session, "player_season_stats", None) or {}
        self.lines: Dict[str, Dict[str, Any]] = {}
        self.minute_cursor: Dict[str, int] = {}
        self.mentions: Dict[str, int] = {}

    # -- lookups -----------------------------------------------------------
    def pidx(self) -> Dict[str, Any]:
        if self._pidx is None:
            try:
                from app.sim_engine.franchise.storyline_engine import _player_index_by_id  # noqa: WPS433

                self._pidx = _player_index_by_id(self.session) or {}
            except Exception:
                self._pidx = {}
        return self._pidx

    def player(self, pid: str) -> Optional[Any]:
        return self.pidx().get(str(pid or ""))

    def team(self, tid: Any) -> Dict[str, Any]:
        return self.teams.get(str(tid or "")) or {}

    def rec(self, tid: Any) -> Dict[str, Any]:
        return self.stand.get(str(tid or "")) or {}

    def rand(self, *parts: Any) -> random.Random:
        return random.Random(_h(self.st.get("salt"), self.iso, self.day_idx, *parts))

    # -- player facts ------------------------------------------------------
    def pinfo(self, pid: str, fallback_name: str = "", team_id: str = "") -> Dict[str, Any]:
        pid = str(pid or "")
        p = self.player(pid)
        row = self.pss.get(pid) if isinstance(self.pss, dict) else None
        row = row if isinstance(row, dict) else {}
        name = ""
        pos = ""
        age = 0
        ovr = 0.0
        cap = 0.0
        yrs: Optional[int] = None
        expiry = ""
        rights = ""
        draft = {}
        if p is not None:
            ident = getattr(p, "identity", None)
            name = str(getattr(ident, "name", "") or getattr(p, "name", "") or "")
            posv = getattr(ident, "position", "") if ident is not None else getattr(p, "position", "")
            pos = str(getattr(posv, "value", posv) or "")
            age = _si(getattr(ident, "age", 0) if ident is not None else getattr(p, "age", 0))
            ovr = _sf(getattr(p, "overall", 0)) or _sf(getattr(p, "_ovr_memo", 0)) * 99.0
            contract = getattr(p, "contract", None)
            if isinstance(contract, dict):
                cap = _sf(contract.get("cap_hit_m") or contract.get("aav_m"))
                yrs = _si(contract.get("years_remaining"), -1)
                yrs = None if yrs < 0 else yrs
                expiry = str(contract.get("expiry_status") or contract.get("rights_status") or "")
            if not cap:
                cap = _sf(getattr(p, "cap_hit_m", 0))
            rights = str(getattr(p, "rights_status", "") or expiry)
            if ident is not None:
                draft = {"year": getattr(ident, "draft_year", None), "round": getattr(ident, "draft_round", None), "pick": getattr(ident, "draft_pick", None)}
        name = name or str(row.get("name") or fallback_name or "")
        pos = pos or str(row.get("position") or "")
        tid = str(team_id or row.get("team_id") or getattr(p, "team_id", "") or "")
        return {
            "id": pid, "name": name, "last": _last(name), "pos": pos.upper()[:2], "age": age, "ovr": ovr,
            "cap": cap, "yrs": yrs, "rights": rights.upper(), "team_id": tid, "draft": draft,
            "gp": _si(row.get("gp")), "g": _si(row.get("g")), "a": _si(row.get("a")), "pts": _si(row.get("pts", _si(row.get("g")) + _si(row.get("a")))),
            "w": _si(row.get("w")), "so": _si(row.get("so")), "saves": _si(row.get("saves")), "ga": _si(row.get("ga", row.get("goalie_ga"))),
            "sa": _si(row.get("shots_against", row.get("goalie_shots_against"))), "ixg": _sf(row.get("ixg")), "sog": _si(row.get("sog")),
            "cf": _sf(row.get("cf")), "ca": _sf(row.get("ca")), "xgf": _sf(row.get("xgf")), "xga": _sf(row.get("xga")),
            "toi": _si(row.get("toi_sec")),
        }

    def star(self, info: Dict[str, Any]) -> float:
        ovr = _sf(info.get("ovr"), 70.0)
        return _clamp(0.7 + max(0.0, ovr - 70.0) / 9.0, 0.6, 3.4)

    def is_goalie(self, info: Dict[str, Any]) -> bool:
        return str(info.get("pos") or "").upper().startswith("G")

    # -- timestamps ---------------------------------------------------------
    def stamp(self, iso: str, lo: int, hi: int, rng: random.Random) -> Tuple[str, str]:
        iso = str(iso or self.iso)[:10]
        base = rng.randint(int(lo), int(hi))
        cur = self.minute_cursor.get(iso)
        last_ts = str(self.st.get("last_ts") or "")
        if last_ts[:10] == iso:
            try:
                lt = int(last_ts[11:13]) * 60 + int(last_ts[14:16])
            except ValueError:
                lt = 0
            if self.mode == "sync" and base <= lt:
                base = lt + rng.randint(2, 18)
        if cur is not None and base == cur:
            base += 1
        self.minute_cursor[iso] = base
        hh = _hhmm(base)
        return iso, hh


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


def _beat_writer(F: _Pass, tid: str) -> Dict[str, Any]:
    tm = F.team(tid)
    r = random.Random(_h(F.st.get("salt"), "beat", tm.get("abbr") or tid))
    first = _pick(r, BEAT_FIRST)
    last = _pick(r, BEAT_LAST)
    outlet = f"{tm.get('city') or 'Local'} {_pick(r, BEAT_OUTLET)}"
    handle_style = r.randrange(3)
    if handle_style == 0:
        handle = f"@{first[0]}{_slug(last)}_{tm.get('abbr') or 'NHL'}"
    elif handle_style == 1:
        handle = f"@{_slug(last)}On{_slug(tm.get('fan') or tm.get('nick') or 'Hockey')}"
    else:
        handle = f"@{first}{_slug(last)}Beat"
    return {
        "id": f"beat_{tm.get('abbr') or tid}", "name": f"{first} {last}", "handle": handle, "type": "beat",
        "badge": "beat", "outlet": outlet, "team_id": str(tid), "color": tm.get("color"), "verified": True,
    }


def _fan_account(F: _Pass, tid: str, persona: Optional[str] = None, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    tm = F.team(tid)
    r = rng or F.rand("fanpick", tid)
    pers = next((p for p in FAN_PERSONAS if p["key"] == persona), None) or _pick(r, FAN_PERSONAS)
    seed = random.Random(_h(F.st.get("salt"), "fan", tm.get("abbr"), pers["key"]))
    fan = tm.get("fan") or tm.get("nick") or "Hockey"
    suffix = _pick(seed, pers["suffixes"])
    first = _pick(seed, ["Dave", "Kyle", "Jess", "Marc", "Steph", "Tyler", "Bri", "Nate", "Ally", "Rob", "Sam", "Jo"])
    num = seed.randint(2, 98)
    style = seed.randrange(3)
    if style == 0:
        handle = f"@{_slug(fan)}{_slug(suffix)}{first}"
        name = f"{fan} {suffix} {first}"
    elif style == 1:
        handle = f"@{_slug(fan)}_{_slug(suffix)}{num}"
        name = f"{fan} {suffix}"
    else:
        handle = f"@{first}{_slug(fan)}Fan{num}"
        name = f"{first} ({fan} {suffix.lower()})"
    return {
        "id": f"fan_{tm.get('abbr') or tid}_{pers['key']}", "name": name, "handle": handle, "type": "fan",
        "badge": "fan", "persona": pers["key"], "bias": pers["bias"], "team_id": str(tid), "color": tm.get("color"), "verified": False,
    }


def _insider(key: str) -> Dict[str, Any]:
    row = INSIDERS.get(key) or INSIDERS["ellison"]
    return {"id": key, "name": row["name"], "handle": row["handle"], "type": "insider", "badge": row.get("badge", "insider"), "outlet": row["outlet"], "verified": True}


def _official() -> Dict[str, Any]:
    return {**OFFICIAL, "verified": True}


def _stats_account(rng: random.Random, prefer: str = "") -> Dict[str, Any]:
    row = next((a for a in STATS_ACCOUNTS if a["id"] == prefer), None) or _pick(rng, STATS_ACCOUNTS)
    return {**row, "type": "stats", "badge": "stats", "verified": False}


def _meme_account(rng: random.Random) -> Dict[str, Any]:
    row = _pick(rng, MEME_ACCOUNTS)
    return {**row, "type": "meme", "badge": "meme", "verified": False}


def _burner_account(rng: random.Random, voice: str = "") -> Dict[str, Any]:
    pool = [a for a in BURNER_ACCOUNTS if not voice or a["voice"] == voice] or BURNER_ACCOUNTS
    row = _pick(rng, pool)
    return {**row, "type": "burner", "badge": "burner", "verified": False}


def _player_account(F: _Pass, info: Dict[str, Any]) -> Dict[str, Any]:
    ent = (getattr(F.session, "universe_players", None) or {}).get(str(info.get("id") or "")) or {}
    handle = str((ent.get("social") or {}).get("handle") or "")
    if not handle:
        handle = f"@{_slug(info.get('name'))}{(_h(info.get('id')) % 89) + 10}"
    tm = F.team(info.get("team_id"))
    return {
        "id": f"player_{info.get('id')}", "name": info.get("name"), "handle": handle, "type": "player", "badge": "player",
        "team_id": str(info.get("team_id") or ""), "color": tm.get("color"), "verified": True, "player_id": str(info.get("id") or ""),
    }


def _agent_account(F: _Pass, pid: str) -> Dict[str, Any]:
    p = F.player(pid)
    prof = getattr(p, "agent_profile", None) if p is not None else None
    name = str((prof or {}).get("name") or "") if isinstance(prof, dict) else ""
    agency = str((prof or {}).get("agency") or "") if isinstance(prof, dict) else ""
    if not name:
        try:
            from app.sim_engine.franchise.storyline_engine import PLAYER_AGENTS  # noqa: WPS433

            row = PLAYER_AGENTS[_h(pid) % len(PLAYER_AGENTS)]
            name, agency = row["name"], row["agency"]
        except Exception:
            name, agency = "Player Agent", "Agency"
    return {"id": f"agent_{_slug(name)}", "name": name, "handle": f"@{_slug(agency) or _slug(name)}", "type": "agent", "badge": "agent", "outlet": agency, "verified": True}


def _reddit_user(rng: random.Random, team: Optional[Dict[str, Any]] = None) -> str:
    if team and rng.random() < 0.55:
        base = f"{_slug(team.get('fan') or team.get('nick'))}{_pick(rng, REDDIT_SUFFIX)}"
    else:
        base = f"{_slug(_pick(rng, REDDIT_PREFIX))}{_pick(rng, REDDIT_SUFFIX)}"
    tail = rng.choice(["", str(rng.randint(2, 99)), str(rng.randint(100, 9999)), "_"])
    return f"u/{base}{tail}"


# ---------------------------------------------------------------------------
# Post / thread construction
# ---------------------------------------------------------------------------

_BASE_LIKES = {"insider": 2400, "official": 1700, "beat": 360, "fan": 60, "stats": 380, "meme": 950, "burner": 210, "player": 4200, "agent": 800, "media": 1300, "gm": 300}


def _engagement(F: _Pass, acct: Dict[str, Any], *, mag: float, star: float, market: float, controversy: float, rng: random.Random) -> Dict[str, int]:
    base = _BASE_LIKES.get(str(acct.get("type") or "fan"), 120)
    noise = rng.lognormvariate(0.0, 0.42)
    likes = max(1, int(base * _clamp(mag, 0.2, 6.0) * _clamp(star, 0.5, 3.6) * _clamp(market, 0.6, 1.8) * noise))
    rp = rng.uniform(0.06, 0.2) + (0.12 if acct.get("type") in ("insider", "official") else 0.0)
    rep = rng.uniform(0.02, 0.07) + _clamp(controversy, 0.0, 1.0) * rng.uniform(0.12, 0.45)
    reposts = int(likes * rp)
    replies = int(likes * rep) + rng.randint(0, 3)
    views = int(likes * rng.uniform(28, 60)) + rng.randint(40, 400)
    return {"likes": likes, "reposts": reposts, "replies": replies, "views": views}


def _add_post(
    F: _Pass,
    acct: Dict[str, Any],
    text: Optional[str],
    *,
    kind: str,
    cat: str,
    when: Tuple[str, str],
    team_ids: Iterable[Any] = (),
    player_id: str = "",
    player_name: str = "",
    attach: Optional[Dict[str, Any]] = None,
    mag: float = 1.0,
    star: float = 1.0,
    sentiment: float = 0.0,
    controversy: float = 0.0,
    knowledge: str = "report",
    platform: str = "twitter",
    priority: float = 1.0,
    storyline_id: str = "",
    related: str = "",
    source_trade_id: str = "",
) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = str(text).strip()
    if len(text) < 8 and not str(kind).startswith(("player", "guarantee")):
        return None
    if not text:
        return None
    tids = [str(t) for t in team_ids if str(t or "") in F.teams]
    if player_id and F.utid and F.utid not in tids:
        p_team = str((F.pss.get(str(player_id)) or {}).get("team_id") or "") if isinstance(F.pss, dict) else ""
        if p_team == F.utid:
            tids.append(F.utid)
    market = max([F.team(t).get("market", 1.0) for t in tids] or [1.0])
    rng = F.rand("eng", kind, text[:40])
    eng = _engagement(F, acct, mag=mag, star=star, market=market, controversy=controversy, rng=rng)
    F.st["seq"] = int(F.st.get("seq") or 0) + 1
    iso, hhmm = when
    post = {
        "v": FEED_VERSION,
        "id": f"sx{F.st['seq']:x}{_h(text) % 4096:03x}",
        "seq": F.st["seq"],
        "platform": platform,
        "kind": kind,
        "cat": cat,
        "author_type": acct.get("type"),
        "author_id": acct.get("id"),
        "author_name": acct.get("name"),
        "handle": acct.get("handle"),
        "badge": acct.get("badge"),
        "outlet": acct.get("outlet") or "",
        "persona": acct.get("persona") or acct.get("voice") or "",
        "verified": bool(acct.get("verified")),
        "author_team_id": str(acct.get("team_id") or ""),
        "author_color": acct.get("color") or (F.team(acct.get("team_id")).get("color") if acct.get("team_id") else None),
        "text": text,
        "calendar_iso": iso,
        "time": hhmm,
        "ts": f"{iso}T{hhmm}",
        "calendar_day": F.day_idx,
        "team_ids": tids,
        "team_id": tids[0] if tids else "",
        "player_id": str(player_id or ""),
        "player_name": str(player_name or ""),
        "sentiment": round(_clamp(sentiment, -1.0, 1.0), 2),
        "knowledge_type": knowledge,
        "heat": int(_clamp(mag * 22 * star, 5, 99)),
        "related_headline": related,
        "storyline_id": storyline_id,
        "source_trade_id": source_trade_id,
        "_prio": priority,
        **eng,
    }
    if attach:
        post["attach"] = attach
    if acct.get("player_id"):
        post["author_player_id"] = acct["player_id"]
    F.posts.append(post)
    if player_id:
        F.mentions[str(player_id)] = F.mentions.get(str(player_id), 0) + 1
    return post


def _comment(author: str, text: Optional[str], up: int, *, flair: str = "", sent: float = 0.0, rival: bool = False, op: bool = False) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    return {"author": author, "text": str(text)[:420], "upvotes": int(up), "flair": flair, "sentiment": round(sent, 2), "is_rival": rival, "is_op": op, "replies": []}


def _add_thread(
    F: _Pass,
    *,
    sub: str,
    title: Optional[str],
    body: str,
    kind: str,
    cat: str,
    flair: str,
    when: Tuple[str, str],
    comments: List[Optional[Dict[str, Any]]],
    team_ids: Iterable[Any] = (),
    player_id: str = "",
    player_name: str = "",
    attach: Optional[Dict[str, Any]] = None,
    mag: float = 1.0,
    sentiment: float = 0.0,
    knowledge: str = "report",
    op_author: str = "",
    storyline_id: str = "",
    source_trade_id: str = "",
) -> Optional[Dict[str, Any]]:
    if not title:
        return None
    tids = [str(t) for t in team_ids if str(t or "") in F.teams]
    rng = F.rand("thread", kind, title[:50])
    clean = [c for c in comments if c]
    if not clean:
        return None
    market = max([F.team(t).get("market", 1.0) for t in tids] or [1.0])
    base = 1500.0 if sub == "r/hockey" else 320.0 * market
    up = int(base * _clamp(mag, 0.3, 6.0) * rng.lognormvariate(0.0, 0.45)) + rng.randint(3, 40)
    spread = sum(abs(c.get("sentiment", 0.0)) for c in clean) / max(1, len(clean))
    mixed = len({(c.get("sentiment", 0.0) > 0.15) - (c.get("sentiment", 0.0) < -0.15) for c in clean}) >= 3
    ratio = 0.95 - (0.18 if mixed else 0.0) - (0.12 if knowledge in ("claim", "speculation") else 0.0) - spread * 0.05
    ratio = round(_clamp(ratio + rng.uniform(-0.04, 0.03), 0.52, 0.99), 2)
    for c in clean:
        c["upvotes"] = int(max(1, c["upvotes"] * (0.35 + up / 1800.0)))
        for rpl in c.get("replies") or []:
            rpl["upvotes"] = int(max(1, rpl["upvotes"] * (0.3 + up / 2600.0)))
    clean.sort(key=lambda c: -c["upvotes"])
    ccount = len(clean) + sum(len(c.get("replies") or []) for c in clean) + int(up * rng.uniform(0.12, 0.4 if mixed else 0.25))
    F.st["seq"] = int(F.st.get("seq") or 0) + 1
    iso, hhmm = when
    thread = {
        "v": FEED_VERSION,
        "thread_id": f"ih{F.st['seq']:x}{_h(title) % 4096:03x}",
        "seq": F.st["seq"],
        "platform": "reddit",
        "subreddit": sub,
        "title": title[:300],
        "body": body[:1800],
        "op_author": op_author or _reddit_user(rng, F.team(tids[0]) if tids else None),
        "flair": flair,
        "kind": kind,
        "cat": cat,
        "upvotes": up,
        "upvote_ratio": ratio,
        "comment_count": ccount,
        "top_comments": clean[:8],
        "calendar_iso": iso,
        "created_at": iso,
        "time": hhmm,
        "ts": f"{iso}T{hhmm}",
        "calendar_day": F.day_idx,
        "team_ids": tids,
        "team_id": tids[0] if tids else "",
        "player_id": str(player_id or ""),
        "player_name": str(player_name or ""),
        "sentiment_score": round(sum(c.get("sentiment", 0.0) for c in clean) / max(1, len(clean)), 3),
        "knowledge_type": knowledge,
        "heat": int(_clamp(mag * 25, 5, 99)),
        "storyline_id": storyline_id,
        "source_trade_id": source_trade_id,
    }
    if attach:
        thread["attach"] = attach
    F.threads.append(thread)
    return thread


def _with_replies(F: _Pass, parent: Optional[Dict[str, Any]], replies: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if parent is None:
        return None
    parent["replies"] = [r for r in replies if r][:3]
    return parent


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


def _score_attach(F: _Pass, g: Dict[str, Any]) -> Dict[str, Any]:
    h, a = F.team(g["hid"]), F.team(g["aid"])
    return {
        "type": "score",
        "home": {"id": g["hid"], "abbr": h.get("abbr"), "name": h.get("full"), "score": g["hs"], "shots": g.get("hsog"), "xg": g.get("hxg"), "rec": F.rec(g["hid"]).get("rec", "")},
        "away": {"id": g["aid"], "abbr": a.get("abbr"), "name": a.get("full"), "score": g["as"], "shots": g.get("asog"), "xg": g.get("axg"), "rec": F.rec(g["aid"]).get("rec", "")},
        "status": "Final" + ("/SO" if g.get("so") else "/OT" if g.get("ot") else ""),
        "stars": [{"player_id": s["id"], "name": s["name"], "line": s["line"], "team_id": s["team_id"]} for s in g.get("stars", [])[:3]],
        "date": g.get("iso"),
    }


def _statline_attach(F: _Pass, info: Dict[str, Any], line: str, season: str = "", label: str = "") -> Dict[str, Any]:
    tm = F.team(info.get("team_id"))
    return {
        "type": "statline",
        "player_id": info.get("id"),
        "name": info.get("name"),
        "pos": info.get("pos"),
        "age": info.get("age"),
        "team_id": info.get("team_id"),
        "abbr": tm.get("abbr"),
        "line": line,
        "season": season,
        "label": label,
        **_headshot_bits(F.player(str(info.get("id") or ""))),
    }


def _headshot_bits(player: Any) -> Dict[str, Any]:
    if player is None:
        return {}
    try:
        from app.sim_engine.generation.player_headshots import merge_headshot_into_row

        # merge_headshot_into_row returns a new dict; it doesn't fill the one passed in.
        row = merge_headshot_into_row({}, player) or {}
        keep = ("avatar_seed", "face_variant", "skin_tone", "hair_style", "hair_color", "facial_hair", "expression", "age_bucket",
                "nationality", "nationality_code", "nhl_id", "nhl_player_id", "real_nhl_import", "portrait_source")
        return {k: v for k, v in row.items() if v not in (None, "") and ("headshot" in k or k in keep)}
    except Exception:
        return {}


def _season_line(info: Dict[str, Any]) -> str:
    if str(info.get("pos") or "").startswith("G"):
        sa = max(1, _si(info.get("sa")))
        sv = (_si(info.get("saves")) / sa) if _si(info.get("sa")) else 0.0
        gaa = (_si(info.get("ga")) * 3600.0 / max(1, _si(info.get("toi")))) if _si(info.get("toi")) else 0.0
        bits = [f"{_si(info.get('w'))} W", f"{_svpct(sv)} SV%" if sv else ""]
        if gaa:
            bits.append(f"{gaa:.2f} GAA")
        if _si(info.get("so")):
            bits.append(f"{_si(info.get('so'))} SO")
        return " · ".join(b for b in bits if b) + f" in {_si(info.get('gp'))} GP"
    return f"{_si(info.get('g'))}G {_si(info.get('a'))}A {_si(info.get('pts'))}P in {_si(info.get('gp'))} GP"


def _contract_attach(info: Dict[str, Any], aav: float, years: int, team_id: str, label: str = "Contract") -> Dict[str, Any]:
    return {
        "type": "contract", "player_id": info.get("id"), "name": info.get("name"), "pos": info.get("pos"), "age": info.get("age"),
        "team_id": team_id, "aav": round(_sf(aav), 3), "years": int(years or 0), "total": round(_sf(aav) * int(years or 0), 2),
        "season": _season_line(info) if info.get("gp") else "", "label": label,
    }


def _standings_attach(F: _Pass, conf: str, focus: Iterable[str] = ()) -> Dict[str, Any]:
    tids = sorted([t for t, r in F.stand.items() if r.get("conf") == conf], key=lambda t: F.stand[t]["conf_rank"])
    focus_set = {str(x) for x in focus}
    rows = []
    for t in tids[4:12] if len(tids) > 10 else tids:
        r = F.stand[t]
        rows.append({"team_id": t, "abbr": F.team(t).get("abbr"), "rank": r["conf_rank"], "pts": r["pts"], "gp": r["gp"], "rec": r["rec"], "odds": round(r.get("odds", 0.5), 3), "focus": t in focus_set, "in": bool(r.get("in"))})
    return {"type": "standings", "conference": conf, "rows": rows}


# ---------------------------------------------------------------------------
# Day context: per-player game lines (diffed from the season ledger)
# ---------------------------------------------------------------------------

_SNAP_KEYS = ("gp", "g", "a", "pts", "sog", "w", "so", "ga", "saves", "shots_against")


def _snapshot(pss: Dict[str, Any]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for pid, row in (pss or {}).items():
        if not isinstance(row, dict):
            continue
        g, a = _si(row.get("g")), _si(row.get("a"))
        vals = [_si(row.get("gp")), g, a, _si(row.get("pts", g + a)), _si(row.get("sog")), _si(row.get("w")),
                _si(row.get("so")), _si(row.get("ga", row.get("goalie_ga"))), _si(row.get("saves")),
                _si(row.get("shots_against", row.get("goalie_shots_against")))]
        out[str(pid)] = vals
    return out


def _day_lines(F: _Pass) -> Dict[str, Dict[str, Any]]:
    """Per-player lines since the last pass (only players whose GP moved)."""
    cur = _snapshot(F.pss)
    prev = F.st.get("snap") or {}
    lines: Dict[str, Dict[str, Any]] = {}
    if prev:
        for pid, vals in cur.items():
            old = prev.get(pid) or [0] * len(_SNAP_KEYS)
            if vals[0] <= old[0]:
                continue
            d = {k: vals[i] - (old[i] if i < len(old) else 0) for i, k in enumerate(_SNAP_KEYS)}
            row = F.pss.get(pid) or {}
            d.update({"id": pid, "name": str(row.get("name") or ""), "team_id": str(row.get("team_id") or ""),
                      "pos": str(row.get("position") or "").upper()[:2]})
            if d["name"]:
                lines[pid] = d
    F.st["snap"] = cur
    # form memory (last 8 games of points / goalie results) for streak posts
    form = F.st.setdefault("form", {})
    for pid, d in lines.items():
        gp = max(1, int(d["gp"]))
        total = int(d["pts"]) if not d["pos"].startswith("G") else int(d["w"])
        per_game = [total // gp + (1 if i < total % gp else 0) for i in range(gp)]
        if d["pos"].startswith("G"):
            per_game = [min(1, v) for v in per_game]
        form[pid] = (list(form.get(pid) or []) + per_game[-3:])[-10:]
    if len(form) > 1600:
        for k in list(form.keys())[: len(form) - 1400]:
            form.pop(k, None)
    return lines


def _game_line_text(d: Dict[str, Any]) -> str:
    if str(d.get("pos") or "").startswith("G"):
        sa = d.get("shots_against") or 0
        return f"{d.get('saves') or max(0, sa - d.get('ga', 0))} saves on {sa}" if sa else "in net"
    bits = []
    if d.get("g"):
        bits.append(_plural(d["g"], "goal"))
    if d.get("a"):
        bits.append(_plural(d["a"], "assist"))
    return ", ".join(bits) or "no points"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _pgt_comment_pool(F: "_Pass", rng: random.Random, *, gd: Dict[str, Any], win: str, lose: str, ws: int, ls: int,
                      skaters: List[Dict[str, Any]], goalies: List[Dict[str, Any]], user_game: bool) -> List[Optional[Dict[str, Any]]]:
    """Comments that react to what actually happened in this game (scoreline, goalies,
    shots, xG, standout lines, the record) instead of a fixed handful of lines."""
    W, Lt = F.team(win), F.team(lose)
    user_won = win == F.utid
    me = F.team(F.utid) if user_game else W
    them = (Lt if user_won else W) if user_game else Lt
    rec = F.rec(F.utid).get("rec", "") if user_game else F.rec(win).get("rec", "")
    my_tid = F.utid if user_game else win
    home_side = my_tid == gd["hid"]
    my_sog, opp_sog = (gd["hsog"], gd["asog"]) if home_side else (gd["asog"], gd["hsog"])
    my_xg, opp_xg = (gd["hxg"], gd["axg"]) if home_side else (gd["axg"], gd["hxg"])
    margin = ws - ls
    happy = user_won if user_game else True
    out: List[Tuple[float, str, float, bool]] = []  # (weight, text, sentiment, rival)

    def add(w: float, txt: Optional[str], sent: float, rival: bool = False) -> None:
        if txt:
            out.append((w, txt, sent, rival))

    top = skaters[0] if skaters and skaters[0]["pts"] else None
    second = skaters[1] if len(skaters) > 1 and skaters[1]["pts"] else None
    if top:
        tl = _last(top["name"])
        add(3, _pick(rng, [f"{top['name']} with {_game_line_text(top)}. Give that man a raise.",
                           f"{tl} is playing the best hockey of his life and I need everyone to notice",
                           f"{_game_line_text(top)} from {tl}. Put some respect on it.",
                           f"{tl} game. That's the comment."]), 0.6)
    if second and top and second["team_id"] == top["team_id"]:
        add(1.5, f"{_last(top['name'])} and {_last(second['name'])} have something going. Keep them together.", 0.4)
    for gk in goalies:
        sv = gk.get("saves", 0)
        ga = gk.get("ga", 0)
        nm = _last(gk["name"])
        mine_g = gk["team_id"] == my_tid
        if gk.get("w") and sv >= 35:
            add(3 if mine_g else 2, _pick(rng, [f"{nm} stole that. {sv} saves. Nobody else in the building deserved two points.",
                                                f"{sv} saves from {nm}. Buy that man dinner.",
                                                f"we got outshot and {nm} said no"]), 0.7, not mine_g)
        if ga >= 5:
            add(2.5, _pick(rng, [f"{ga} goals against. {nm} didn't get much help but man that was rough",
                                 f"pull {nm} after the 4th one, what are we doing",
                                 f"{nm} looked like he was seeing beach balls... going the other way"]), -0.6, not mine_g)
    if margin >= 4:
        add(2.5, (_pick(rng, ["that was a statement game", "beat them like a drum. what a night", f"{margin}-goal win and it didn't feel that close"])
                  if happy else _pick(rng, ["embarrassing. turned it off after the second", f"lost by {margin}. I want my three hours back", "that's a team meeting kind of loss"])),
            0.7 if happy else -0.8)
    if gd.get("ot") or gd.get("so"):
        add(2, _pick(rng, ["my heart can't take OT games like that", "3-on-3 is pure chaos and I love it", "shootouts should be illegal",
                           "loser point is a participation trophy, change my mind"]), 0.1)
    if my_sog and opp_sog and opp_sog - my_sog >= 10:
        add(2, _pick(rng, [f"outshot {opp_sog}-{my_sog}. that's not sustainable",
                           f"{opp_sog} shots against. the D got caved in all night"]), -0.4)
    if my_xg and opp_xg and abs(my_xg - opp_xg) >= 1.0:
        better = my_xg > opp_xg
        verdict = ("deserved it" if better else "got lucky tbh") if happy else ("we got robbed" if better else "deserved to lose honestly")
        add(1.5, f"xG was {my_xg:.1f}-{opp_xg:.1f} for what it's worth — {verdict}", 0.0)
    try:
        _w, _l = [int(x) for x in str(rec).split("-")[:2]]
        good_year = _w >= _l
    except Exception:
        good_year = True
    if user_game:
        if user_won and good_year:
            pool = ["Not a pretty win but two points are two points", f"{rec}. the vibes are immaculate",
                    "playoffs? don't jinx it", "coach has them playing the right way lately"]
        elif user_won:
            pool = [f"nice win, still {rec} though", "a win! in this economy!", "enjoy it, these have been rare",
                    "tank commander would not approve of this W"]
        else:
            pool = [f"We're {rec} and somehow it feels worse", "This team is going to give me grey hair",
                    "PP has to be better than this", "same story every night: good first period then nothing",
                    "fire the PP coach, I'm serious"]
        add(2, _pick(rng, pool), 0.4 if user_won else -0.5)
        add(1.2, _pick(rng, [f"{them.get('nick')} fans in here acting like they won the Cup", f"never liked the {them.get('nick')}, never will",
                             f"{them.get('nick')} have no business beating us"]), -0.1)
        add(1.0, _pick(rng, ["Refs were brutal both ways", "the camera work on the broadcast was worse than our D",
                             "who's ready for the game day thread tomorrow", "beer was $14. still went. no regrets"]), 0.0)
    else:
        add(1.5, f"{Lt.get('nick')} fans, you okay?", -0.1, True)
        add(1.2, _pick(rng, [f"{W.get('nick')} are quietly a problem this year", f"the {Lt.get('nick')} need a trade, yesterday",
                             "Fun game to watch honestly"]), 0.1)
    rng.shuffle(out)
    out.sort(key=lambda t: -t[0] * rng.uniform(0.6, 1.4))
    recent = set(_recent_fps(F.st)[-160:])
    fresh = [o for o in out if _fingerprint(o[1]) not in recent]
    picked = (fresh + [o for o in out if o not in fresh])[:6]
    for _o in picked:
        _remember_fp(F.st, _o[1])
    comments: List[Optional[Dict[str, Any]]] = []
    for i, (w, txt, sent, rival) in enumerate(picked):
        team_for_user = them if rival else me
        c = _comment(_reddit_user(rng, team_for_user), txt, int(rng.randint(15, 140) * (1.0 + w)), sent=sent, rival=rival)
        if c and i < 2 and rng.random() < 0.55:
            reply = _pick(rng, ["this", "underrated comment", "hard agree", "nah you're overreacting", "every. single. game.",
                                "username checks out", "say it louder for the coaching staff"])
            c = _with_replies(F, c, [_comment(_reddit_user(rng, me), reply, rng.randint(3, 60), sent=sent * 0.5)])
        comments.append(c)
    return comments


def _fan_flavor_templates(F: _Pass, tid: str, happy: bool) -> List[str]:
    """Lines that only make sense for this club right now (record, coach, market)."""
    tm, r = F.team(tid), F.rec(tid)
    out: List[str] = []
    city = tm.get("city") or ""
    if happy:
        if r.get("odds", 0.5) >= 0.8:
            out += ["{rec}. start planning the parade route, " + city + " (I'm kidding) (I'm not kidding)"]
        if r.get("odds", 0.5) <= 0.25:
            out += ["a win is nice but the lottery odds just took a hit. I am a complicated person",
                    "{score} W in a lost season. we take these and we don't ask questions"]
        if tm.get("coach"):
            out += [f"{_last(tm['coach'])} pushed the right buttons tonight. there, I said something nice"]
        if tm.get("market", 1.0) >= 1.25:
            out += ["the radio call-in shows are going to be unbearable tomorrow and I love it"]
    else:
        if tm.get("coach_hot") or tm.get("coach_security", 0.5) < 0.35:
            out += [f"how many more of these before {_last(tm.get('coach') or 'the coach')} gets the call",
                    "the coach is out of answers. you can see it on the bench"]
        if r.get("odds", 0.5) >= 0.7:
            out += ["one loss. everyone breathe. we're fine. (are we fine?)"]
        if r.get("odds", 0.5) <= 0.25:
            out += ["at this point I'm watching for the draft lottery percentages",
                    "{rec}. I'm learning the names of draft prospects instead of our power play units"]
        if tm.get("market", 1.0) >= 1.25:
            out += ["the local paper is going to have a field day with this one. {score}."]
    return out


def _gen_games(F: _Pass, games: List[Dict[str, Any]], lines: Dict[str, Dict[str, Any]]) -> None:
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for d in lines.values():
        by_team.setdefault(d["team_id"], []).append(d)
    for g in games:
        hid, aid = str(g.get("home_id")), str(g.get("away_id"))
        if hid not in F.teams or aid not in F.teams:
            continue
        hs, as_ = _si(g.get("home_score", g.get("home_goals"))), _si(g.get("away_score", g.get("away_goals")))
        win, lose = (hid, aid) if hs > as_ else (aid, hid)
        ws, ls = max(hs, as_), min(hs, as_)
        ot = bool(g.get("overtime") or g.get("ot"))
        so = bool(g.get("shootout"))
        players = by_team.get(hid, []) + by_team.get(aid, [])
        skaters = sorted([p for p in players if not p["pos"].startswith("G")], key=lambda p: (p["pts"], p["g"]), reverse=True)
        goalies = [p for p in players if p["pos"].startswith("G") and p.get("shots_against")]
        stars = []
        for p in skaters[:3]:
            if p["pts"]:
                stars.append({"id": p["id"], "name": p["name"], "line": _game_line_text(p), "team_id": p["team_id"]})
        for gk in goalies:
            if gk.get("w") and gk.get("saves", 0) >= 30 and len(stars) < 3:
                stars.append({"id": gk["id"], "name": gk["name"], "line": _game_line_text(gk), "team_id": gk["team_id"]})
        gd = {"hid": hid, "aid": aid, "hs": hs, "as": as_, "ot": ot and not so, "so": so, "hsog": _si(g.get("home_sog")), "asog": _si(g.get("away_sog")),
              "hxg": round(_sf(g.get("home_xg")), 2), "axg": round(_sf(g.get("away_xg")), 2), "stars": stars, "iso": F.iso}
        att = _score_attach(F, gd)
        W, Lt = F.team(win), F.team(lose)
        rng = F.rand("game", g.get("game_id"))
        user_game = F.utid in (hid, aid)
        mag = 0.8 + 0.3 * (ws - ls >= 4) + 0.4 * user_game + 0.2 * ot
        when = F.stamp(F.iso, 22 * 60 + 5, 23 * 60 + 40, rng)
        tag = " (SO)" if so else " (OT)" if ot else ""
        win_top = next((p for p in skaters if p["team_id"] == win and p["pts"]), None)
        win_gk = next((gk for gk in goalies if gk["team_id"] == win and gk.get("w") and gk.get("saves", 0) >= 30), None)
        if win_gk and (not win_top or win_gk.get("ga", 9) == 0 or (win_top.get("pts", 0) <= 1 and win_gk.get("saves", 0) >= 42)):
            star_txt = f" {win_gk['name']} made {win_gk.get('saves', 0)} saves."
        elif win_top:
            star_txt = f" {win_top['name']} led the way with {_game_line_text(win_top)}."
        else:
            star_txt = ""
        _add_post(F, _official(), f"FINAL{tag}: {W.get('abbr')} {ws}, {Lt.get('abbr')} {ls}.{star_txt} {W.get('abbr')} move to {F.rec(win).get('rec', '')}.",
                  kind="final", cat="game", when=when, team_ids=[win, lose], attach=att, mag=mag, knowledge="confirmed")
        # Beat writers for both clubs
        for tid in (win, lose):
            won = tid == win
            mine = [p for p in skaters if p["team_id"] == tid]
            top = mine[0] if mine and mine[0]["pts"] else None
            tm = F.team(tid)
            opp = F.team(lose if won else win)
            my_shots = _si(g.get("home_sog" if tid == hid else "away_sog"))
            opp_shots = _si(g.get("away_sog" if tid == hid else "home_sog"))
            shot_note = None
            if my_shots and opp_shots:
                if my_shots - opp_shots >= 8:
                    shot_note = f"outshot the {opp.get('nick')} {my_shots}-{opp_shots}"
                elif opp_shots - my_shots >= 8:
                    shot_note = f"won it despite being outshot {opp_shots}-{my_shots}"
            ctx = {"abbr": tm.get("abbr"), "opp": opp.get("nick"), "score": f"{ws}-{ls}", "rec": F.rec(tid).get("rec"),
                   "top": top["name"] if top else None, "line": _game_line_text(top) if top else None,
                   "shots": my_shots or None, "shot_note": shot_note, "coach": tm.get("coach") or None}
            tmpl = (["{abbr} beat the {opp} {score}. {top}: {line}. Record now {rec}.",
                     "{abbr} improve to {rec} with a {score} win over the {opp}. {top} ({line}) drove it.",
                     "{abbr} {shot_note} and take it {score}. {top}: {line}.",
                     "{abbr} take it {score} over the {opp}. {coach} on {top}: \"He was the difference tonight.\""]
                    if won else
                    ["{abbr} drop a {score} decision to the {opp}. {rec} on the year.",
                     "Tough night: {abbr} fall {score} to the {opp} despite {shots} shots. {top} had {line}.",
                     "{coach} after the {score} loss to the {opp}: \"Not good enough. We know it.\" {abbr} now {rec}."])
            if tm.get("is_user") or rng.random() < 0.55:
                _add_post(F, _beat_writer(F, tid), _compose_fresh(F.st, rng, tmpl, ctx), kind="recap", cat="game",
                          when=F.stamp(F.iso, 22 * 60 + 20, 23 * 60 + 55, rng), team_ids=[tid], player_id=top["id"] if top else "",
                          player_name=top["name"] if top else "", mag=0.7 + 0.5 * tm.get("is_user", False), knowledge="confirmed")
            # Fans
            if tm.get("is_user") or rng.random() < 0.35:
                fan = _fan_account(F, tid, rng=rng)
                bias = fan.get("bias", 0.0)
                happy = won
                fctx = {"fan": tm.get("fan"), "opp": opp.get("nick"), "top": top["last"] if top and "last" in top else (_last(top["name"]) if top else None),
                        "score": f"{ws}-{ls}", "rec": F.rec(tid).get("rec")}
                tmpl = (["{top} is HIM. {fan} win {score}, I'm not taking questions",
                         "{score} over the {opp}. {rec}. we are so back",
                         "nobody is talking about how {top} has carried this team. {score} W",
                         "beat the {opp} {score} and I'm being normal about it (I am not being normal about it)",
                         "{top} said 'not tonight' to the entire {opp} roster",
                         "if you didn't watch {fan} hockey tonight that's on you. {score}",
                         "{rec}. a team of destiny? no. a team of {top}? maybe",
                         "the {opp} came into our building and left with nothing. {score} 🚨"]
                        if happy else
                        ["{rec}. I've seen enough. something has to change",
                         "lost {score} to the {opp}. at least the beer was cold",
                         "how do you lose to the {opp}. HOW. {fan} fans deserve better",
                         "the {opp} aren't even good and we made them look like the '77 Habs",
                         "{score}. I'm going to bed. don't @ me until there's a trade",
                         "{rec} and the GM is still 'evaluating'. evaluate THIS",
                         "every loss to the {opp} takes a year off my life. {score}",
                         "started the game with hope, ended it with a {score} L and a headache"])
                tmpl = tmpl + _fan_flavor_templates(F, tid, happy)
                if happy and bias < -0.2:
                    tmpl = ["won {score} but the process is still bad and you all know it", "fine. {score}. wake me up when it matters",
                            "a {score} win doesn't fix the roster. I will be back here complaining by Thursday",
                            "{score} W. enjoy it. I'm still not buying it"]
                _add_post(F, fan, _compose_fresh(F.st, rng, tmpl, fctx), kind="fan_react", cat="game", when=F.stamp(F.iso, 22 * 60 + 30, 23 * 60 + 59, rng),
                          team_ids=[tid], mag=0.6 + 0.4 * tm.get("is_user", False), sentiment=0.6 if happy else -0.6, controversy=0.2 if not happy else 0.05)
        # Milestones
        for p in skaters[:4]:
            if p["g"] >= 3:
                info = F.pinfo(p["id"], p["name"], p["team_id"])
                hrng = F.rand("hat", p["id"])
                abbr_p = F.team(p["team_id"]).get("abbr")
                ht_author = _beat_writer(F, p["team_id"]) if hrng.random() < 0.6 else _insider(_pick(hrng, ["knox", "lee", "ellison"]))
                ht_text = _pick_fresh(F.st, hrng, [
                    f"HAT TRICK: {p['name']} ({abbr_p}) with three tonight. That's {info['g']} goals in {info['gp']} games this season.",
                    f"Three for {p['name']}. {_ordinal(max(1, info['g']))}-goal season pace aside, that's his {'first' if info['g'] <= 4 else 'latest'} hat trick of the year. {info['g']} G in {info['gp']} GP.",
                    f"{p['name']} completes the hat trick for {abbr_p}. Hats are raining down. {info['g']} goals on the season.",
                    f"{_last(p['name'])} with a natural feel for the net tonight: three goals, {p['sog']} shots. {abbr_p} fans are out of hats.",
                ])
                _add_post(F, ht_author, ht_text,
                          kind="hat_trick", cat="game", when=F.stamp(F.iso, 21 * 60 + 40, 23 * 60, rng), team_ids=[p["team_id"]], player_id=p["id"], player_name=p["name"],
                          attach=_statline_attach(F, info, _season_line(info), label="Season"), mag=2.0, star=F.star(info), knowledge="confirmed")
                _add_post(F, _meme_account(rng), _pick_fresh(F.st, rng, [f"hats on the ice for {_last(p['name'])} 🎩🎩🎩 someone check on the {Lt.get('nick') if p['team_id'] == win else W.get('nick')} goalie",
                          f"{_last(p['name'])} said 'one hat? no. three.' 🎩🎩🎩",
                          f"the {Lt.get('nick') if p['team_id'] == win else W.get('nick')} goalie is going to see {_last(p['name'])} in his sleep tonight"]),
                          kind="meme", cat="game", when=F.stamp(F.iso, 22 * 60, 23 * 60 + 50, rng), team_ids=[p["team_id"]], player_id=p["id"], player_name=p["name"], mag=1.4)
            elif p["pts"] >= 4:
                info = F.pinfo(p["id"], p["name"], p["team_id"])
                _add_post(F, _stats_account(rng), f"{p['name']} tonight: {_game_line_text(p)}. Season: {_season_line(info)}.",
                          kind="big_night", cat="game", when=F.stamp(F.iso, 22 * 60, 23 * 60 + 50, rng), team_ids=[p["team_id"]], player_id=p["id"],
                          player_name=p["name"], attach=_statline_attach(F, info, _season_line(info), label="Season"), mag=1.4, star=F.star(info))
        for gk in goalies:
            if gk.get("w") and gk.get("ga", 0) == 0 and gk.get("saves", 0) >= 15:
                info = F.pinfo(gk["id"], gk["name"], gk["team_id"])
                _add_post(F, _insider("howe"), f"Shutout for {gk['name']}: {gk['saves']} saves as {F.team(gk['team_id']).get('abbr')} blank the {(Lt if gk['team_id'] == win else W).get('nick')}. {_season_line(info)}.",
                          kind="shutout", cat="game", when=F.stamp(F.iso, 22 * 60, 23 * 60 + 30, rng), team_ids=[gk["team_id"]], player_id=gk["id"], player_name=gk["name"],
                          attach=_statline_attach(F, info, _season_line(info), label="Season"), mag=1.5, star=F.star(info), knowledge="confirmed")
        # xG heist
        wxg = gd["hxg"] if win == hid else gd["axg"]
        lxg = gd["axg"] if win == hid else gd["hxg"]
        if lxg - wxg >= 1.4:
            _add_post(F, _stats_account(rng, "pdowatch"), f"{W.get('abbr')} won {ws}-{ls} while losing the xG battle {wxg:.2f}-{lxg:.2f}. Goaltending or luck, but not a repeatable process.",
                      kind="xg_heist", cat="analytics", when=F.stamp(F.iso, 23 * 60, 23 * 60 + 59, rng), team_ids=[win, lose], mag=0.9, controversy=0.3)
        # Reddit: PGT for the user's games and big results
        if user_game or ws - ls >= 5 or any(p["g"] >= 3 for p in skaters):
            sub = F.team(F.utid).get("sub") if user_game else "r/hockey"
            user_won = win == F.utid
            F.team(F.utid if user_game else win)
            top = skaters[0] if skaters else None
            comments = _pgt_comment_pool(F, rng, gd=gd, win=win, lose=lose, ws=ws, ls=ls,
                                         skaters=skaters, goalies=goalies, user_game=user_game)
            _add_thread(F, sub=sub or "r/hockey", title=f"Post Game Thread: {W.get('full')} {ws}, {Lt.get('full')} {ls}{tag}",
                        body=f"Final{tag}: {W.get('abbr')} {ws} - {Lt.get('abbr')} {ls}. Shots {gd['hsog']}-{gd['asog']}. " + (" | ".join(f"{s['name']}: {s['line']}" for s in stars) if stars else ""),
                        kind="pgt", cat="game", flair="Post Game Thread", when=F.stamp(F.iso, 22 * 60 + 15, 23 * 60 + 30, rng), comments=comments,
                        team_ids=[win, lose], attach=att, mag=1.2 + 0.5 * user_game, sentiment=0.4 if user_won else -0.3, knowledge="confirmed")


def _gen_daily_thread(F: _Pass, games: List[Dict[str, Any]], lines: Dict[str, Dict[str, Any]]) -> None:
    if len(games) < 3:
        return
    rng = F.rand("daily")
    results = []
    for g in games:
        h, a = F.team(str(g.get("home_id"))), F.team(str(g.get("away_id")))
        hs, as_ = _si(g.get("home_score", g.get("home_goals"))), _si(g.get("away_score", g.get("away_goals")))
        ot = " (OT)" if g.get("overtime") or g.get("ot") else ""
        results.append(f"{a.get('abbr')} {as_} @ {h.get('abbr')} {hs}{ot}")
    best = sorted([d for d in lines.values() if not d["pos"].startswith("G")], key=lambda d: (d["pts"], d["g"]), reverse=True)[:3]
    comments = [_comment(_reddit_user(rng, F.team(b["team_id"])), f"{b['name']} tonight: {_game_line_text(b)}. Underrated player in this league.", rng.randint(30, 250), sent=0.5) for b in best[:2]]
    comments += [_comment(_reddit_user(rng), _pick(rng, ["Every night I watch hockey and every night the refs find a new way", "Wild night around the league", "My parlay died on the first game again", "Standings are getting spicy"]), rng.randint(10, 160), sent=0.0),
                 _comment(_reddit_user(rng, F.team(F.utid)), f"{F.team(F.utid).get('fan')} fans checking in, {F.rec(F.utid).get('rec', '')} and still believing", rng.randint(5, 90), sent=0.2)]
    _add_thread(F, sub="r/hockey", title=f"Daily Discussion - {_pretty_date(F.iso)}: {len(games)} games tonight", body="Results: " + " | ".join(results),
                kind="daily", cat="game", flair="Daily Discussion", when=F.stamp(F.iso, 23 * 60 + 30, 23 * 60 + 58, rng), comments=comments,
                team_ids=[], mag=1.0, knowledge="confirmed", op_author="u/HockeyMod")


def _gen_streaks(F: _Pass, lines: Dict[str, Dict[str, Any]]) -> None:
    form = F.st.get("form") or {}
    posted = 0
    for pid, d in sorted(lines.items(), key=lambda kv: -kv[1].get("pts", 0)):
        if posted >= 2:
            break
        if d["pos"].startswith("G"):
            continue
        f = form.get(pid) or []
        info = F.pinfo(pid, d["name"], d["team_id"])
        rng = F.rand("streak", pid)
        if len(f) >= 6 and all(x > 0 for x in f[-6:]) and not _flag(F.st, f"hot:{pid}:{info['gp'] // 6}"):
            _set_flag(F.st, f"hot:{pid}:{info['gp'] // 6}")
            _add_post(F, _stats_account(rng), f"{d['name']} has a point in {sum(1 for _ in f[-6:])}+ straight ({sum(f[-6:])} points). Season: {_season_line(info)}.",
                      kind="hot_streak", cat="player", when=F.stamp(F.iso, 9 * 60, 13 * 60, rng), team_ids=[d["team_id"]], player_id=pid, player_name=d["name"],
                      attach=_statline_attach(F, info, _season_line(info), label="Season"), mag=1.1, star=F.star(info))
            posted += 1
        elif len(f) >= 8 and sum(f[-8:]) == 0 and info["ovr"] >= 80 and not _flag(F.st, f"cold:{pid}:{info['gp'] // 8}"):
            _set_flag(F.st, f"cold:{pid}:{info['gp'] // 8}")
            tm = F.team(d["team_id"])
            _add_post(F, _fan_account(F, d["team_id"], "doomer", rng), f"{info['last']} pointless in 8 straight. at {_money(info['cap'])} a year?? {tm.get('fan')} deserve better" if info["cap"] else f"{info['last']} pointless in 8 straight. someone wake him up",
                      kind="slump", cat="player", when=F.stamp(F.iso, 10 * 60, 15 * 60, rng), team_ids=[d["team_id"]], player_id=pid, player_name=d["name"],
                      mag=1.0, star=F.star(info), sentiment=-0.7, controversy=0.5)
            posted += 1


def _gen_injuries(F: _Pass) -> None:
    rows, seen = _seen(F.st, "inj")
    for row in list(getattr(F.session, "injury_log_all", None) or [])[-60:]:
        key = str(row.get("id") or "")
        if not key or key in seen or str(row.get("calendar_iso") or "")[:10] > F.iso:
            continue
        _mark_seen(F.st, "inj", key)
        if _iso_days_between(str(row.get("calendar_iso") or F.iso), F.iso) > 3:
            continue
        tid, pid = str(row.get("team_id") or ""), str(row.get("player_id") or "")
        info = F.pinfo(pid, str(row.get("player_name") or ""), tid)
        if not info["name"]:
            continue
        games = _si(row.get("games_initial") or row.get("games"))
        sev = str(row.get("severity") or row.get("tier") or "").lower()
        rng = F.rand("inj", key)
        out = "week-to-week" if games <= 10 else "month-to-month" if games <= 30 else "out long-term"
        major = games >= 15 or sev in ("major", "severe", "season_ending")
        acct = _insider("lee") if major or info["ovr"] >= 82 else _beat_writer(F, tid)
        _add_post(F, acct, f"{F.team(tid).get('abbr')} injury update: {info['name']} is {out} (est. {games} games). {_season_line(info) if info['gp'] else ''}".strip(),
                  kind="injury", cat="injury", when=F.stamp(F.iso, 10 * 60, 18 * 60, rng), team_ids=[tid], player_id=pid, player_name=info["name"],
                  mag=1.0 + major, star=F.star(info), sentiment=-0.5, knowledge="confirmed")
        if major and info["ovr"] >= 78:
            tm = F.team(tid)
            _add_thread(F, sub=tm.get("sub") or "r/hockey", title=f"[{acct['name']}] {info['name']} expected to miss ~{games} games", body=f"Huge blow for the {tm.get('nick')}. {_season_line(info)}.",
                        kind="injury", cat="injury", flair="News", when=F.stamp(F.iso, 12 * 60, 19 * 60, rng), team_ids=[tid], player_id=pid, player_name=info["name"],
                        comments=[_comment(_reddit_user(rng, tm), "Season's over. Pack it up.", rng.randint(50, 300), sent=-0.8),
                                  _comment(_reddit_user(rng, tm), "Next man up. Time for the kids to show something.", rng.randint(30, 200), sent=0.2),
                                  _comment(_reddit_user(rng), f"{tm.get('nick')} need to call someone up from the AHL now", rng.randint(10, 90), sent=-0.1)],
                        mag=1.4, sentiment=-0.5, knowledge="confirmed")


def _gen_roster_moves(F: _Pass) -> None:
    """CPU call-ups / send-downs (services/cpu_roster_moves.py) on the beat writers' feeds."""
    rows, seen = _seen(F.st, "crm")
    posted = 0
    for mv in list(getattr(F.session, "cpu_roster_moves_log", None) or [])[-40:]:
        key = str(mv.get("id") or "")
        if not key or key in seen:
            continue
        _mark_seen(F.st, "crm", key)
        if posted >= 5 or int(mv.get("day") or 0) < F.day_idx - 3:
            continue
        tid = str(mv.get("team_id") or "")
        if tid not in F.teams:
            continue
        up, down = mv.get("up") or {}, mv.get("down") or {}
        rng = F.rand("crm", key)
        abbr = F.team(tid).get("abbr")
        method = str(mv.get("method") or "")
        if method == "goalie_assigned":
            text = f"{abbr} assign G {up.get('name')} to the AHL. Back to a two-goalie rotation."
        elif method == "goalie_waived":
            text = f"{abbr} have placed G {up.get('name')} on waivers."
        elif method == "recalled" or not down:
            text = f"{abbr} recall {up.get('name')} from the AHL."
        elif method == "waived":
            text = f"{abbr} have placed {down.get('name')} on waivers and recalled {up.get('name')} from the AHL."
        else:
            text = _pick(rng, [f"{abbr} recall {up.get('name')} from the AHL; {down.get('name')} assigned down.",
                               f"Roster move: {up.get('name')} up, {down.get('name')} down for {abbr}.",
                               f"{up.get('name')} gets the call to {abbr}. {down.get('name')} heads to the AHL."])
        _add_post(F, _beat_writer(F, tid), text, kind="roster_move", cat="news", when=F.stamp(F.iso, 10 * 60, 17 * 60, rng),
                  team_ids=[tid], player_id=str(up.get("player_id") or ""), player_name=str(up.get("name") or ""), mag=0.6, knowledge="confirmed")
        posted += 1


def _gen_fa_signings(F: _Pass) -> None:
    """Free-agent signings (CPU and user) on the feed: insider, fans, the player."""
    rows, seen = _seen(F.st, "fa_sign")
    posted = 0
    for s in list((getattr(F.session, "cpu_fa_signings", None) or {}).get("signings") or [])[-80:]:
        if not isinstance(s, dict):
            continue
        pid = str(s.get("player_id") or "")
        tid = str(s.get("team_id") or "")
        key = f"{F.season}:{pid}:{tid}"
        if not pid or key in seen:
            continue
        _mark_seen(F.st, "fa_sign", key)
        seen.add(key)
        if tid not in F.teams or posted >= 24:
            continue
        tm = F.team(tid)
        info = F.pinfo(pid, str(s.get("name") or ""), team_id=tid)
        name = info["name"] or str(s.get("name") or "")
        if not name:
            continue
        aav = _sf(s.get("aav_m"))
        yrs = _si(s.get("years"), 1)
        ovr = _sf(s.get("overall")) or _sf(info.get("ovr"))
        rng = F.rand("fa_sign", key)
        star = _clamp(0.7 + max(0.0, ovr - 70.0) / 9.0, 0.6, 3.4)
        user = tid == F.utid
        when = F.stamp(F.iso, 9 * 60, 22 * 60, rng)
        text = _pick(rng, [
            f"SIGNED: {tm.get('abbr')} land {name} on a {_plural(yrs, 'year')} deal, {_money(aav)} AAV.",
            f"Done deal: {name} to the {tm.get('nick')}. {yrs}x{_money(aav)}.",
            f"Source: {name} has agreed with {tm.get('abbr')}. Term {yrs}, AAV {_money(aav)}.",
        ])
        _add_post(F, _insider(_pick(rng, ["ellison", "vargas", "reid"])), text, kind="fa_signing", cat="signing", when=when,
                  team_ids=[tid], player_id=pid, player_name=name, mag=0.9 + 0.5 * user + max(0.0, ovr - 80) * 0.08,
                  star=star, knowledge="confirmed")
        posted += 1
        if ovr >= 80 or user:
            fan = _fan_account(F, tid, "homer", rng)
            pool = [
                f"{info['last'] or name} in {tm.get('fan')} colours. love it",
                f"{_money(aav)} for {info['last'] or name}? fair price honestly",
                f"GM cooked with this one. welcome {info['last'] or name}",
                f"not sure about {yrs} years but the player is legit",
            ]
            if aav >= 7.0:
                pool.append(f"{_money(aav)} a year... he better be a difference maker")
            _add_post(F, fan, _pick(rng, pool), kind="fa_reaction", cat="signing",
                      when=F.stamp(F.iso, 10 * 60, 23 * 60, rng), team_ids=[tid], player_id=pid, player_name=name,
                      mag=0.5 + 0.2 * user, sentiment=0.5)
        if ovr >= 82 or user:
            acct = _player_account(F, {**info, "team_id": tid})
            _add_post(F, acct, _pick(rng, [
                f"Excited to join the {tm.get('nick')}. Can't wait to get to work.",
                f"New chapter. Thank you to everyone who got me here. Let's go {tm.get('fan')}!",
                f"Grateful for the opportunity in {tm.get('city')}. See you at camp.",
            ]), kind="fa_player", cat="signing", when=F.stamp(F.iso, 12 * 60, 23 * 60, rng),
                      team_ids=[tid], player_id=pid, player_name=name, mag=0.7 + 0.3 * user, star=star, sentiment=0.8)


def _gen_trades(F: _Pass) -> None:
    league = getattr(getattr(F.session, "sim", None), "league", None)
    rows, seen = _seen(F.st, "trade")
    for tr in list(getattr(league, "trade_history", None) or [])[-30:]:
        key = str(tr.get("trade_id") or "")
        if not key or key in seen or not tr.get("accepted", True):
            continue
        _mark_seen(F.st, "trade", key)
        if _si(tr.get("calendar_day"), F.day_idx) < F.day_idx - 3:
            continue
        moved = [m for m in (tr.get("moved_players") or []) if isinstance(m, dict)]
        picks = [m for m in (tr.get("moved_picks") or []) if isinstance(m, dict)]
        teams = [str(t) for t in (tr.get("participating_teams") or [])]
        if len(teams) < 2:
            continue
        rng = F.rand("trade", key)
        parts = []
        for tid in teams:
            got = [m.get("player_name") for m in moved if str(m.get("acquiring_team_id")) == tid and m.get("player_name")]
            gp = [m for m in picks if str(m.get("acquiring_team_id") or m.get("to_team_id")) == tid]
            bits = list(got)
            if gp:
                bits.append(_plural(len(gp), "pick"))
            if bits:
                parts.append(f"{F.team(tid).get('abbr')} get: {', '.join(bits)}")
        if not parts:
            continue
        user_inv = F.utid in teams
        star_names = [m for m in moved if m.get("asset_id")]
        best = max((F.pinfo(str(m.get("asset_id")), str(m.get("player_name") or "")) for m in star_names), key=lambda i: i["ovr"], default=None)
        star = F.star(best) if best else 1.0
        mag = 1.4 + 0.8 * user_inv + 0.6 * (str(tr.get("importance") or "") in ("major", "blockbuster"))
        when = F.stamp(F.iso, 11 * 60, 20 * 60, rng)
        reason = str(tr.get("reason_text") or "").strip()
        if reason and not reason.endswith((".", "!", "?")):
            reason = reason.rsplit(";", 1)[0].rsplit(",", 1)[0].strip() + "."
        _add_post(F, _insider(_pick(rng, ["ellison", "vargas"])), "TRADE: " + " | ".join(parts) + (f". {reason[:150]}" if reason else ""),
                  kind="trade", cat="trade", when=when, team_ids=teams, player_id=best["id"] if best else "", player_name=best["name"] if best else "",
                  mag=mag, star=star, knowledge="confirmed", source_trade_id=key)
        if best and best["cap"]:
            _add_post(F, _insider("reid"), f"Cap side of the {F.team(teams[0]).get('abbr')}/{F.team(teams[1]).get('abbr')} deal: {best['name']} carries {_money(best['cap'])}" + (f" for {_plural(best['yrs'], 'more year')}." if best.get("yrs") is not None else "."),
                      kind="trade_cap", cat="trade", when=F.stamp(F.iso, 12 * 60, 21 * 60, rng), team_ids=teams, player_id=best["id"], player_name=best["name"], mag=0.9, star=star)
        loser = str(tr.get("lopsided_loser_team_id") or "")
        for tid in teams:
            tm = F.team(tid)
            angry = tid == loser
            fan = _fan_account(F, tid, "doomer" if angry else "homer", rng)
            rel = "fractured" in reason.lower() or "demand" in reason.lower() or "wish" in reason.lower()
            new_guy = best["last"] if best else "the new guy"
            if angry:
                pool = [f"what did we just do. {tm.get('fan')} front office fleeced again", "I need someone to explain this trade to me slowly",
                        f"we gave up THAT for THIS? {tm.get('fan')} twitter is in shambles", "this trade will age like milk. screenshot it"]
            elif rel:
                pool = ["good riddance honestly. the room needed this", "addition by subtraction. moving on", "wish him well but this had to happen",
                        f"the drama is someone else's problem now. welcome {new_guy}"]
            else:
                pool = [f"LOVE this move. {tm.get('fan')} got better today", f"ok I'm in on this. {new_guy} is going to fit perfectly",
                        "fine trade. not a home run, not a disaster", f"{new_guy} jersey already in the cart", "need to see him play before I judge but I like the vibes",
                        f"didn't see that coming. {new_guy}? sure, why not"]
            txt = _pick_fresh(F.st, rng, pool)
            _add_post(F, fan, txt, kind="trade_react", cat="trade", when=F.stamp(F.iso, 12 * 60, 23 * 60, rng), team_ids=[tid], mag=0.7 + 0.6 * tm.get("is_user", False),
                      sentiment=-0.7 if angry else 0.5, controversy=0.5 if angry else 0.1, source_trade_id=key)
        _add_thread(F, sub="r/hockey", title=f"[{_insider('ellison')['name']}] " + "; ".join(parts), body=str(tr.get("reason_text") or tr.get("headline") or ""),
                    kind="trade", cat="trade", flair="Trade", when=F.stamp(F.iso, 12 * 60, 21 * 60, rng), team_ids=teams, player_id=best["id"] if best else "",
                    comments=[_comment(_reddit_user(rng, F.team(teams[0])), f"{F.team(teams[0]).get('abbr')} win this trade easily", rng.randint(40, 400), sent=0.4),
                              _comment(_reddit_user(rng, F.team(teams[1])), f"Disagree. {F.team(teams[1]).get('nick')} got the best player in the deal", rng.randint(30, 300), sent=0.2, rival=True),
                              _comment(_reddit_user(rng), "Need to see the retained salary before I judge", rng.randint(10, 150)),
                              _comment(_reddit_user(rng), "This league is so back. Trades every week", rng.randint(5, 80), sent=0.3)],
                    mag=mag, sentiment=0.0, knowledge="confirmed", source_trade_id=key)


def _gen_storylines(F: _Pass) -> None:
    rows, seen = _seen(F.st, "story")
    n = 0
    for ev in list(getattr(F.session, "storyline_events", None) or [])[-40:]:
        key = str(ev.get("id") or ev.get("storyline_id") or "")
        if not key or key in seen:
            continue
        _mark_seen(F.st, "story", key)
        if str(ev.get("calendar_iso") or "")[:10] < _iso_shift(F.iso, -2) or n >= 6:
            continue
        heat = _si(ev.get("heat"))
        head = str(ev.get("headline") or "").strip()
        if heat < 28 or len(head) < 12:
            continue
        n += 1
        tid = str(ev.get("team_id") or ev.get("team") or "")
        rng = F.rand("story", key)
        acct = {"id": f"rep_{ev.get('reporter_id')}", "name": str(ev.get("reporter_name") or "League Wire"), "handle": f"@{_slug(ev.get('reporter_name') or 'LeagueWire')}",
                "type": "beat", "badge": "media", "outlet": str(ev.get("outlet_name") or ""), "verified": True} if ev.get("reporter_name") else _beat_writer(F, tid)
        _add_post(F, acct, head[:280], kind="story", cat="rumor" if str(ev.get("knowledge_type")) in ("rumor", "speculation", "claim") else "news",
                  when=F.stamp(F.iso, 8 * 60, 21 * 60, rng), team_ids=[tid], player_id=str(ev.get("player_id") or ""), player_name=str(ev.get("player_name") or ""),
                  mag=0.6 + heat / 50.0, knowledge=str(ev.get("knowledge_type") or "report"), storyline_id=str(ev.get("storyline_id") or ""), related=head)


def _gen_demands(F: _Pass) -> None:
    demands = getattr(F.session, "trade_demands", None) or {}
    open_d = [d for d in demands.values() if isinstance(d, dict) and str(d.get("status") or "open") in ("open", "active", "formal", "requested")]
    for d in open_d:
        pid = str(d.get("player_id") or "")
        if _flag(F.st, f"demand:{d.get('demand_id')}"):
            continue
        _set_flag(F.st, f"demand:{d.get('demand_id')}")
        info = F.pinfo(pid, str(d.get("player_name") or ""), str(d.get("team_id") or ""))
        r2 = F.rand("demand", pid)
        _add_post(F, _insider("ellison"), f"Hearing {info['name']} has asked {F.team(info['team_id']).get('abbr')} for a trade. Main issue: {d.get('primary_complaint') or 'his role'}.",
                  kind="trade_request", cat="rumor", when=F.stamp(F.iso, 9 * 60, 20 * 60, r2), team_ids=[info["team_id"]], player_id=pid, player_name=info["name"],
                  mag=1.6, star=F.star(info), knowledge="report", controversy=0.6)


def _gen_burner_rumors(F: _Pass) -> None:
    """Burner rumor mill: sellers, buyers and hot seats (once per played day)."""
    rng = F.rand("burner")
    cands = []
    for tid, r in F.stand.items():
        if r.get("gp", 0) >= 15 and r.get("odds", 0.5) < 0.25:
            cands.append(("seller", tid))
        elif r.get("gp", 0) >= 15 and r.get("odds", 0.5) > 0.8:
            cands.append(("buyer", tid))
    rng.shuffle(cands)
    for kind, tid in cands[: rng.randint(1, 3)]:
        tm = F.team(tid)
        roster = [p for p in (F.pss or {}).values() if isinstance(p, dict) and str(p.get("team_id")) == tid and _si(p.get("gp")) > 5]
        if not roster:
            continue
        p = _pick(rng, sorted(roster, key=lambda r: -_si(r.get("pts")))[:8])
        name = str(p.get("name") or "")
        if kind == "seller":
            txt = _pick(rng, [f"{tm.get('abbr')} taking calls on basically everyone not named {name}. heard it from two different scouts",
                              f"source in the {tm.get('city')} press box says {tm.get('coach') or 'the coach'} is coaching for his job this week",
                              f"don't be shocked if {name} is moved before the deadline. {tm.get('abbr')} front office is listening"])
        else:
            txt = _pick(rng, [f"{tm.get('abbr')} sniffing around for a top-4 D. cap room is the problem, not the will",
                              f"{tm.get('abbr')} want to go all in this year. {name} reportedly lobbying management for help up front",
                              f"multiple teams asked about {tm.get('abbr')}'s 1st this year. they're keeping it... for now"])
        _add_post(F, _burner_account(rng), txt, kind="burner_rumor", cat="rumor", when=F.stamp(F.iso, 13 * 60, 23 * 60 + 50, rng), team_ids=[tid], player_name=name,
                  mag=0.7, knowledge="claim", controversy=0.5, platform="burner")


def _gen_standings(F: _Pass) -> None:
    if not F.stand or F.day_idx % 4 != 0:
        return
    rng = F.rand("standings")
    confs = sorted({r.get("conf") for r in F.stand.values() if r.get("conf")})
    for conf in confs:
        tids = sorted([t for t, r in F.stand.items() if r.get("conf") == conf], key=lambda t: F.stand[t]["conf_rank"])
        if len(tids) < 9 or F.stand[tids[0]]["gp"] < 10:
            continue
        e, n = F.stand[tids[7]], F.stand[tids[8]]
        _add_post(F, _stats_account(rng, "netfront"), f"{conf} wildcard race: {F.team(tids[7]).get('abbr')} hold the last spot at {e['pts']} pts; {F.team(tids[8]).get('abbr')} {abs(e['pts'] - n['pts'])} back with {n['gp']} GP. "
                  f"Playoff odds: {F.team(tids[7]).get('abbr')} {_pct(e.get('odds', 0.5), 0)}, {F.team(tids[8]).get('abbr')} {_pct(n.get('odds', 0.5), 0)}.",
                  kind="race", cat="standings", when=F.stamp(F.iso, 9 * 60, 11 * 60, rng), team_ids=[tids[7], tids[8]], attach=_standings_attach(F, conf, [tids[7], tids[8]]), mag=1.0)
    ur = F.rec(F.utid)
    if ur.get("gp", 0) >= 10:
        tm = F.team(F.utid)
        _add_post(F, _beat_writer(F, F.utid), f"{tm.get('abbr')} sit {_ordinal(ur.get('conf_rank', 0))} in the {ur.get('conf')} at {ur.get('rec')} ({ur.get('pts')} pts). Playoff odds: {_pct(ur.get('odds', 0.5), 0)}. Last 10: {ur.get('l10') or '—'}.",
                  kind="user_race", cat="standings", when=F.stamp(F.iso, 8 * 60, 10 * 60, rng), team_ids=[F.utid], mag=0.9)


def _gen_governance(F: _Pass) -> None:
    gov = getattr(F.session, "league_governance", None) or {}
    rows, seen = _seen(F.st, "gov")
    for h in list(gov.get("history") or [])[-10:]:
        key = str(h.get("proposal_id") or "")
        if not key or key in seen:
            continue
        _mark_seen(F.st, "gov", key)
        rng = F.rand("gov", key)
        passed = h.get("status") == "passed"
        _add_post(F, _official(), f"Board of Governors: \"{h.get('title')}\" {'PASSED' if passed else 'FAILED'} {h.get('yes')}-{h.get('no')}.",
                  kind="gov_vote", cat="news", when=F.stamp(F.iso, 13 * 60, 16 * 60, rng), mag=1.2, knowledge="confirmed")
        _add_thread(F, sub="r/hockey", title=f"Board of Governors votes {'in' if passed else 'down'}: {h.get('title')} ({h.get('yes')}-{h.get('no')})", body=str(h.get("summary") or ""),
                    kind="gov", cat="news", flair="League News", when=F.stamp(F.iso, 13 * 60, 18 * 60, rng),
                    comments=[_comment(_reddit_user(rng), _pick(rng, ["Owners voting for owners, shocking", "Honestly a good change for the league", "This will be repealed in two years"]), rng.randint(40, 400), sent=0.0),
                              _comment(_reddit_user(rng), _pick(rng, ["Small markets get squeezed again", "Finally some common sense", "Who even asked for this"]), rng.randint(20, 200), sent=-0.2)],
                    mag=1.3, knowledge="confirmed")


def _gen_carryover(F: _Pass) -> None:
    """Season-opening regression watch from last season's luck (season_carryover)."""
    carry = getattr(F.session, "team_luck_carryover", None) or {}
    if not carry or _flag(F.st, f"carry:{F.season}"):
        return
    rows = [(t, r) for t, r in carry.items() if isinstance(r, dict) and int(r.get("season") or 0) == F.season - 1 and t in F.teams]
    if not rows:
        return
    _set_flag(F.st, f"carry:{F.season}")
    rng = F.rand("carry")
    rows.sort(key=lambda tr: -(tr[1].get("luck_goals") or 0))
    lucky, unlucky = rows[:3], rows[-3:][::-1]
    fmt = lambda tr: f"{F.team(tr[0]).get('abbr')} ({tr[1].get('luck_goals'):+.0f} goals vs xG, PDO {tr[1].get('pdo')})"  # noqa: E731
    _add_post(F, _stats_account(rng, "pdowatch"), "Regression watch for the new season. Ran hot last year: " + ", ".join(fmt(r) for r in lucky)
              + ". Due for better luck: " + ", ".join(fmt(r) for r in unlucky) + ".", kind="regression", cat="analytics",
              when=F.stamp(F.iso, 9 * 60, 12 * 60, rng), team_ids=[t for t, _ in lucky + unlucky], mag=1.3, controversy=0.3)
    u = carry.get(F.utid)
    if isinstance(u, dict) and int(u.get("season") or 0) == F.season - 1:
        tm = F.team(F.utid)
        verdict = "they'll need to be better, not just luckier" if (u.get("luck_goals") or 0) > 8 else "the underlying numbers say better days are coming" if (u.get("luck_goals") or 0) < -8 else "last year's results matched the process"
        _add_post(F, _beat_writer(F, F.utid), f"{tm.get('abbr')} finished last season with {u.get('pts')} pts, {u.get('xgf_pct')}% xG share and a {u.get('pdo')} PDO. Short version: {verdict}.",
                  kind="carryover", cat="analytics", when=F.stamp(F.iso, 10 * 60, 13 * 60, rng), team_ids=[F.utid], mag=1.0)


# ---------------------------------------------------------------------------
# Day runner / catch-up
# ---------------------------------------------------------------------------

def _gen_player_voices(F: _Pass, games: List[Dict[str, Any]], lines: Dict[str, Dict[str, Any]]) -> None:
    from services.player_social_engine import gen_player_voices  # noqa: WPS433

    gen_player_voices(F, games, lines)


def _gen_player_event_voices(F: _Pass) -> None:
    from services.player_social_engine import gen_player_event_voices  # noqa: WPS433

    gen_player_event_voices(F)


def _player_social_summary(session: Any) -> Dict[str, Any]:
    try:
        from services.player_social_engine import player_social_summary  # noqa: WPS433

        return player_social_summary(session)
    except Exception:
        _log.exception("player social summary failed")
        return {}


def _iso_for_day(session: Any, day_idx: int) -> str:
    try:
        from services.franchise_sim import _calendar_iso_for_day

        return str(_calendar_iso_for_day(session, int(day_idx)) or "")[:10]
    except Exception:
        return ""


def _current_day(session: Any) -> int:
    return _si(getattr(session, "calendar_cursor", 0))


def _prune(session: Any, today_iso: str) -> None:
    cutoff = _iso_shift(today_iso, -ARCHIVE_DAYS)
    posts = [p for p in list(getattr(session, "social_posts", None) or []) if isinstance(p, dict) and int(p.get("v") or 0) == FEED_VERSION and str(p.get("calendar_iso") or "") >= cutoff]
    threads = [t for t in list(getattr(session, "reddit_threads", None) or []) if isinstance(t, dict) and int(t.get("v") or 0) == FEED_VERSION and str(t.get("calendar_iso") or "") >= cutoff]
    session.social_posts = posts[-MAX_POSTS:]
    session.reddit_threads = threads[-MAX_THREADS:]


def _games_on_day(session: Any, day_idx: int) -> List[Dict[str, Any]]:
    """Every final on ``day_idx`` (scans back from the newest result, bug S9)."""
    out: List[Dict[str, Any]] = []
    season = _si(getattr(session, "season_calendar_year", 0))
    for g in reversed(list(getattr(session, "game_results", None) or [])):
        if not isinstance(g, dict):
            continue
        gs = _si(g.get("season_calendar_year"), season)
        if gs and season and gs != season:
            break
        gd = _si(g.get("calendar_day"), -1)
        if gd == int(day_idx):
            out.append(g)
        elif 0 <= gd < int(day_idx) - 1 and out:
            break
    out.reverse()
    return out


_DAY_GENERATORS = ("games", "daily", "streaks", "injuries", "trades", "moves", "signings", "stories", "demands", "rumors", "standings", "gov", "carry", "players")
_EVENT_GENERATORS = ("injuries", "trades", "moves", "signings", "stories", "demands", "gov", "carry")


def run_social_feed_day(session: Any, day_idx: int, *, lines: Optional[Dict[str, Dict[str, Any]]] = None, events_only: bool = False) -> int:
    """Generate one calendar day of posts.

    A full pass is for a day whose games are final; it advances ``last_day``. An
    ``events_only`` pass covers the current, unplayed day (trades, signings, offseason
    news) and never marks the day as done (bug S1)."""
    iso = _iso_for_day(session, day_idx)
    if not iso:
        return 0
    F = _Pass(session, iso=iso, day_idx=day_idx, mode="events" if events_only else "day")
    games: List[Dict[str, Any]] = [] if events_only else _games_on_day(session, day_idx)
    if lines is None:
        lines = {} if events_only else _day_lines(F)
    gens = {
        "games": (_gen_games, (F, games, lines)), "daily": (_gen_daily_thread, (F, games, lines)), "streaks": (_gen_streaks, (F, lines)),
        "injuries": (_gen_injuries, (F,)), "trades": (_gen_trades, (F,)), "moves": (_gen_roster_moves, (F,)),
        "signings": (_gen_fa_signings, (F,)), "stories": (_gen_storylines, (F,)),
        "demands": (_gen_demands, (F,)), "rumors": (_gen_burner_rumors, (F,)), "standings": (_gen_standings, (F,)),
        "gov": (_gen_governance, (F,)), "carry": (_gen_carryover, (F,)), "players": (_gen_player_voices, (F, games, lines)),
    }
    for key in (_EVENT_GENERATORS if events_only else _DAY_GENERATORS):
        fn, args = gens[key]
        try:
            fn(*args)
        except Exception:
            _log.exception("social feed generator %s failed", getattr(fn, "__name__", fn))
    if events_only:
        try:
            _gen_player_event_voices(F)
        except Exception:
            _log.exception("social feed player event voices failed")
    F.posts.sort(key=lambda p: p["ts"])
    session.social_posts = list(getattr(session, "social_posts", None) or []) + F.posts
    session.reddit_threads = list(getattr(session, "reddit_threads", None) or []) + F.threads
    if not events_only:
        F.st["last_day"] = max(_si(F.st.get("last_day"), -1), int(day_idx))
    if F.posts:
        F.st["last_ts"] = max(str(F.st.get("last_ts") or ""), F.posts[-1]["ts"])
    _prune(session, iso)
    return len(F.posts) + len(F.threads)


def _season_sync(session: Any, st: Dict[str, Any]) -> None:
    season = _si(getattr(session, "season_calendar_year", 0))
    if st.get("season") != season:
        st["season"] = season
        st["last_day"] = -1
        cur = _snapshot(getattr(session, "player_season_stats", None) or {})
        # A fresh ledger (opening night) diffs against nothing, so night one gets its lines.
        fresh = all((v[0] if v else 0) <= 1 for v in cur.values())
        st["snap"] = {} if fresh or not st.get("bootstrapped") else cur
    if not st.get("bootstrapped"):
        # First run on an existing save: seed the stat snapshot so day lines start clean,
        # and treat everything already played as covered.
        cur = _snapshot(getattr(session, "player_season_stats", None) or {})
        fresh = all((v[0] if v else 0) <= 1 for v in cur.values())
        st["snap"] = {} if fresh else cur
        st["bootstrapped"] = True
        st["fmt3"] = True
        st["last_day"] = -1 if fresh else max(_si(st.get("last_day"), -1), _current_day(session) - 1)


def _migrate_cursor(st: Dict[str, Any], first_unplayed: int) -> None:
    """Saves from before the S1 fix marked the unplayed day as done; step back once."""
    if not st.get("fmt3"):
        st["fmt3"] = True
        st["last_day"] = min(_si(st.get("last_day"), -1), int(first_unplayed) - 1)


def _run_played_span(session: Any, st: Dict[str, Any], start: int, end: int) -> int:
    """Full passes for played days ``start..end``. One day (the normal case, called right
    after the sim finishes it) gets exact lines; a longer gap splits one ledger diff by
    each team's game days so a player's totals never land on a single night (bug S3)."""
    if end < start:
        return 0
    probe = _Pass(session, iso=_iso_for_day(session, end) or "", day_idx=end)
    all_lines = _day_lines(probe)
    if start == end:
        return run_social_feed_day(session, end, lines=all_lines)
    games_by_team: Dict[str, List[int]] = {}
    for d in range(start, end + 1):
        for g in _games_on_day(session, d):
            for t in (str(g.get("home_id")), str(g.get("away_id"))):
                games_by_team.setdefault(t, []).append(d)
    per_day: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for pid, ln in all_lines.items():
        days = games_by_team.get(str(ln.get("team_id"))) or []
        if not days:
            continue
        if int(ln.get("gp") or 0) <= 1:
            per_day.setdefault(days[-1], {})[pid] = ln
            continue
        # Several games in the gap: only the per-game average is known, so post it as
        # an average on the last game day instead of inventing a monster night.
        gp = int(ln["gp"])
        avg = dict(ln)
        for k in ("g", "a", "pts", "sog", "w", "so", "ga", "saves", "shots_against"):
            avg[k] = int(round(float(ln.get(k) or 0) / gp))
        avg["gp"] = 1
        avg["_span_avg"] = True
        per_day.setdefault(days[-1], {})[pid] = avg
    made = 0
    for d in range(start, end + 1):
        made += run_social_feed_day(session, d, lines=per_day.get(d, {}))
    return made


def social_feed_after_day(session: Any, day_idx: int) -> int:
    """Sim hook: call once a calendar day's games are final (bug S1/S2/S11 — the feed
    keeps up with the sim instead of catching up when the tab is opened)."""
    with _LOCK:
        st = _state(session)
        _season_sync(session, st)
        _migrate_cursor(st, int(day_idx))
        last = _si(st.get("last_day"), -1)
        if int(day_idx) <= last:
            return 0
        start = max(last + 1, int(day_idx) - ARCHIVE_DAYS + 1)
        return _run_played_span(session, st, start, int(day_idx))


def ensure_social_feed_current(session: Any, max_days: int = ARCHIVE_DAYS) -> int:
    """Cheap catch-up: covers any played day the sim hook missed, then an events-only
    pass for the current (unplayed) day so trades and offseason news still show up."""
    with _LOCK:
        st = _state(session)
        _season_sync(session, st)
        today = _current_day(session)
        _migrate_cursor(st, today)
        last = _si(st.get("last_day"), -1)
        made = 0
        played_end = today - 1
        if last < played_end:
            made += _run_played_span(session, st, max(last + 1, played_end - max(1, int(max_days)) + 1), played_end)
        made += run_social_feed_day(session, today, events_only=True)
        return made


def _match_tab(item: Dict[str, Any], tab: str, utid: str) -> bool:
    if tab in ("", "all"):
        return True
    if tab == "mine":
        return utid in (item.get("team_ids") or [])
    if tab == "rumors":
        return item.get("cat") == "rumor" or str(item.get("knowledge_type") or "") in ("claim", "speculation", "rumor")
    if tab == "games":
        return item.get("cat") in ("game", "analytics", "standings")
    if tab == "trades":
        return item.get("cat") == "trade"
    if tab == "fa":
        return item.get("cat") == "signing" or item.get("kind") in ("fa_signing", "fa_reaction", "fa_player")
    if tab == "players":
        return item.get("cat") == "players" or item.get("author_type") == "player"
    return True


def build_social_feed_response(session: Any, *, tab: str = "all", sub: str = "all", page: int = 0, page_size: int = 40,
                               thread_page: Optional[int] = None, author: str = "") -> Dict[str, Any]:
    try:
        ensure_social_feed_current(session)
    except Exception:
        _log.exception("social feed catch-up failed")
    tpage = page if thread_page is None else max(0, int(thread_page))
    utid = str(getattr(session, "user_team_id", "") or "")
    posts = [p for p in list(getattr(session, "social_posts", None) or []) if isinstance(p, dict) and int(p.get("v") or 0) == FEED_VERSION]
    threads = [t for t in list(getattr(session, "reddit_threads", None) or []) if isinstance(t, dict) and int(t.get("v") or 0) == FEED_VERSION]
    posts = [p for p in posts if _match_tab(p, tab, utid)]
    if author:
        posts = [p for p in posts if str(p.get("author_player_id") or "") == author or str(p.get("author_id") or "") == author]
    all_threads = [t for t in threads if _match_tab(t, tab, utid)]
    threads = [t for t in all_threads if sub in ("", "all") or str(t.get("subreddit") or "").lower() == sub.lower()]
    posts.sort(key=lambda p: (str(p.get("ts") or ""), _si(p.get("seq"))), reverse=True)
    threads.sort(key=lambda t: (str(t.get("ts") or ""), _si(t.get("upvotes"))), reverse=True)
    lo, hi = page * page_size, (page + 1) * page_size
    tlo, thi = tpage * page_size, (tpage + 1) * page_size
    subs = sorted({str(t.get("subreddit") or "") for t in all_threads if t.get("subreddit")})
    pub = lambda rows: [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]  # noqa: E731
    user_sub = (_team_info(session).get(utid) or {}).get("sub") or ""
    return {
        "puckr": pub(posts[lo:hi]),
        "icehole": pub(threads[tlo:thi]),
        "has_more_posts": len(posts) > hi,
        "has_more_threads": len(threads) > thi,
        "thread_page": tpage,
        "user_subreddit": user_sub,
        "player_social": _player_social_summary(session),
        "total_posts": len(posts),
        "total_threads": len(threads),
        "subreddits": subs,
        "today": _iso_for_day(session, _current_day(session)),
        "page": page,
    }


def publish_external_post(
    session: Any,
    *,
    handle: str,
    name: str,
    text: str,
    kind: str = "burner",
    sentiment: float = 0.0,
    controversy: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Drop a post written outside the generators (e.g. the GM's burner) into today's feed."""
    day_idx = _current_day(session)
    iso = _iso_for_day(session, day_idx) or str(getattr(session, "calendar_iso", "") or "")[:10]
    if not iso:
        return None
    F = _Pass(session, iso=iso, day_idx=day_idx, mode="external")
    acct = {"id": f"ext_{_slug(handle)}", "name": name, "handle": handle, "type": "burner", "badge": "anon", "voice": "burner"}
    rng = F.rand("ext", handle, text[:30])
    post = _add_post(
        F, acct, text, kind=kind, cat="rumor", when=F.stamp(iso, 9 * 60, 23 * 60, rng),
        team_ids=[F.utid], mag=0.9 + controversy, sentiment=sentiment, controversy=controversy, knowledge="claim",
    )
    if post:
        session.social_posts = list(getattr(session, "social_posts", None) or []) + F.posts
        _prune(session, iso)  # bug S10
    return post
