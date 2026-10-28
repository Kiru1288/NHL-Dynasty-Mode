"""PuckGPT, history layer: questions about what happened in this save.

Everything here reads the Legacy Hall (season rows, career lines, trades, drafts,
playoff series, awards) plus the live season. The judging rules are fixed and
stated in every answer:

  value      points for skaters, two per win for goalies
  trades     what each side's incoming players produced for that club afterwards
  drafts     current rating against what the pick slot usually becomes
  playoffs   archived series, regular-season points as the seed

Questions the save cannot know (rejected offers, true counterfactuals, game
order inside a series on older saves) get the nearest real answer and say so.
``match`` runs before the other layers, on the raw question.
"""

from __future__ import annotations

import copy
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .answers import _answer, _team_label
from .data import LiveView
from .derived import draft_expectation

Answer = Dict[str, Any]
VALUE = "Value is points for skaters and two per win for goalies."
TRADE_RULE = "Trades are judged on what each side's incoming players produced for that club afterwards (" + VALUE[9:-1] + ")."
DRAFT_RULE = "Picks are judged on current rating against what that slot usually becomes (#1 ≈ 92, #32 ≈ 84, #100 ≈ 78)."
MAJOR = ("hart", "vezina", "norris", "calder", "art ross", "rocket", "conn smythe", "selke", "ted lindsay", "lindsay")
DEADLINE_MONTHS = (2, 3)


# =========================================================================== reading the save
class Hist:
    """One read of the Legacy Hall, cached on the live view."""

    def __init__(self, view: LiveView) -> None:
        from services.legacy_hall import ensure_legacy

        self.view = view
        self.session = view.session
        self.lg = ensure_legacy(view.session)
        self.user = view.user_tid
        self.seasons: Dict[int, Dict[str, Any]] = {
            int(k): s for k, s in (self.lg.get("seasons") or {}).items() if s.get("standings")}
        self.players: Dict[str, Dict[str, Any]] = self.lg.get("players") or {}
        # In the offseason the live ledger still holds the season that was just archived.
        phase = str(getattr(view.session, "season_phase", "") or "")
        self.live = phase != "offseason" and view.season not in self.seasons
        self._lines: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._trades: Optional[List[Dict[str, Any]]] = None
        self._drafts: Optional[Dict[str, Dict[str, Any]]] = None

    # ---- names
    def name(self, pid: str) -> str:
        return str((self.players.get(pid) or {}).get("name") or self.view.row(pid).get("name") or "Unknown player")

    def abbr(self, tid: Any) -> str:
        return self.view.team_meta.get(str(tid or ""), {}).get("abbr") or str(tid or "—")

    def team(self, tid: Any) -> str:
        return _team_label(self.view, str(tid)) if tid else "an unrecorded club"

    def tid_of(self, key: Any) -> str:
        key = str(key or "")
        if key in self.view.team_meta:
            return key
        for tid, meta in self.view.team_meta.items():
            if key and key.upper() == meta.get("abbr", "").upper():
                return tid
        return ""

    def goalie(self, pid: str) -> bool:
        pos = (self.players.get(pid) or {}).get("position") or self.view.row(pid).get("pos") or ""
        return str(pos).upper().startswith("G")

    @property
    def span(self) -> str:
        if not self.seasons:
            return "no finished seasons yet"
        a, b = min(self.seasons), max(self.seasons)
        n = len(self.seasons)
        return f"{n} finished season{'s' if n != 1 else ''} ({_season(a)}{'' if a == b else ' to ' + _season(b)})"

    # ---- careers
    def lines(self, pid: str, scope: str = "regular") -> List[Dict[str, Any]]:
        key = (pid, scope)
        if key in self._lines:
            return self._lines[key]
        pr = self.players.get(pid) or {}
        out = [s for s in pr.get("seasons") or [] if s.get("source") == "simulated" and not s.get("aggregate")
               and (s.get("scope") or "regular") == scope]
        if scope == "regular" and self.live and pid in self.view.players:
            r = self.view.row(pid)
            if r.get("gp"):
                out = out + [{"season": self.view.season, "team_id": r["team_id"], "gp": r["gp"], "g": r["g"], "a": r["a"],
                              "pts": r["pts"], "w": r["w"], "so": r["so"], "pim": r["pim"], "plus_minus": r["plus_minus"],
                              "ovr": r["ovr"], "live": True}]
        self._lines[key] = out
        return out

    def val(self, pid: str, ln: Dict[str, Any]) -> int:
        if self.goalie(pid):
            return 2 * int(ln.get("w") or 0)
        return int(ln.get("pts") or 0)

    def by_season(self, pid: str, scope: str = "regular") -> Dict[int, Dict[str, int]]:
        """Season totals across team splits."""
        out: Dict[int, Dict[str, int]] = {}
        for ln in self.lines(pid, scope):
            s = out.setdefault(int(ln.get("season") or 0), {"gp": 0, "pts": 0, "g": 0, "w": 0, "val": 0, "pim": 0, "pm": 0})
            s["gp"] += int(ln.get("gp") or 0)
            s["pts"] += int(ln.get("pts") or 0)
            s["g"] += int(ln.get("g") or 0)
            s["w"] += int(ln.get("w") or 0)
            s["pim"] += int(ln.get("pim") or 0)
            s["pm"] += int(ln.get("plus_minus") or 0)
            s["val"] += self.val(pid, ln)
        return out

    def career(self, pid: str) -> Dict[str, Any]:
        bs = self.by_season(pid)
        teams: List[str] = []
        for ln in sorted(self.lines(pid), key=lambda x: int(x.get("season") or 0)):
            t = str(ln.get("team_id") or "")
            if t and t not in teams:
                teams.append(t)
        return {"gp": sum(s["gp"] for s in bs.values()), "pts": sum(s["pts"] for s in bs.values()),
                "val": sum(s["val"] for s in bs.values()), "seasons": len(bs), "teams": teams, "by_season": bs}

    def nhl_imported(self, pid: str) -> Dict[str, int]:
        """Real NHL seasons before the save, where the import fetched them."""
        rows = [s for s in (self.players.get(pid) or {}).get("imported_seasons") or []
                if str(s.get("league") or "NHL").upper() in ("NHL", "NATIONAL HOCKEY LEAGUE")]
        seasons = {str(s.get("season")) for s in rows}
        return {"gp": sum(int(s.get("gp") or 0) for s in rows), "pts": sum(int(s.get("pts") or 0) for s in rows),
                "w": sum(int(s.get("w") or 0) for s in rows), "seasons": len(seasons),
                "teams": len({str(s.get("team") or "") for s in rows if s.get("team")})}

    def ovr_now(self, pid: str) -> Optional[float]:
        return self.view.row(pid).get("ovr") if pid in self.view.players else None

    def ovr_path(self, pid: str) -> List[Tuple[int, float]]:
        path = {int(ln.get("season") or 0): float(ln["ovr"]) for ln in self.lines(pid) if ln.get("ovr")}
        now = self.ovr_now(pid)
        if now:
            path[max(path) + 1 if path and self.view.season in path else self.view.season] = float(now)
        return sorted(path.items())

    def awards_of(self, pid: str, major_only: bool = True) -> List[Dict[str, Any]]:
        out = [a for a in (self.players.get(pid) or {}).get("awards") or [] if a.get("place") in (None, 1, "1")]
        if major_only:
            out = [a for a in out if any(m in str(a.get("award") or "").lower() for m in MAJOR)]
        return out

    def cups_of(self, pid: str) -> List[int]:
        return list((self.players.get(pid) or {}).get("championships") or [])

    def retired(self) -> List[str]:
        return [pid for pid, pr in self.players.items() if pr.get("retired_season")]

    # ---- standings & playoffs
    def pts(self, sy: int, tid: str) -> Optional[int]:
        row = (self.seasons.get(sy, {}).get("standings") or {}).get(str(tid))
        return int(row.get("pts") or 0) if row else None

    def rank(self, sy: int, tid: str) -> Optional[int]:
        st = self.seasons.get(sy, {}).get("standings") or {}
        order = sorted(st, key=lambda t: (-int(st[t].get("pts") or 0), -(int(st[t].get("gf") or 0) - int(st[t].get("ga") or 0))))
        return order.index(str(tid)) + 1 if str(tid) in order else None

    def series(self) -> List[Tuple[int, Dict[str, Any]]]:
        return [(sy, ser) for sy, s in sorted(self.seasons.items()) for ser in s.get("playoff_series") or s.get("series") or []
                if ser.get("winner_id")]

    def final_round(self, sy: int) -> int:
        return max([int(s.get("round_index") or 0) for s in self.seasons.get(sy, {}).get("playoff_series") or []] or [0])

    def champions(self) -> List[Tuple[int, str]]:
        return [(sy, str(s["champion_id"])) for sy, s in sorted(self.seasons.items()) if s.get("champion_id")]

    def finalist(self, sy: int) -> Optional[str]:
        s = self.seasons.get(sy) or {}
        if s.get("finalist_id"):
            return str(s["finalist_id"])
        fr = self.final_round(sy)
        for ser in s.get("playoff_series") or []:
            if int(ser.get("round_index") or 0) == fr and fr:
                ids = [str(ser.get("team_high_id")), str(ser.get("team_low_id"))]
                return next((t for t in ids if t != str(s.get("champion_id"))), None)
        return None

    # ---- trades
    def trades(self) -> List[Dict[str, Any]]:
        if self._trades is not None:
            return self._trades
        out = []
        for t in (self.lg.get("trades") or {}).values():
            teams = [str(x) for x in t.get("teams") or []]
            if len(teams) < 2:
                continue
            sy = int(t.get("season") or 0)
            got = {x: 0 for x in teams}
            recv: Dict[str, List[Tuple[Optional[str], str, int]]] = {x: [] for x in teams}
            for mp in t.get("players") or []:
                pid, to = str(mp.get("player_id") or ""), str(mp.get("to_team_id") or "")
                if to not in got:
                    continue
                v = sum(self.val(pid, ln) for ln in self.lines(pid) if str(ln.get("team_id")) == to and int(ln.get("season") or 0) >= sy)
                v += sum(self.val(pid, ln) for ln in self.lines(pid, "playoffs") if str(ln.get("team_id")) == to and int(ln.get("season") or 0) >= sy)
                got[to] += v
                recv[to].append((pid, str(mp.get("name") or self.name(pid)), v))
            for pk in t.get("picks") or []:
                to = str(pk.get("to_team_id") or "")
                if to in recv:
                    label = pk.get("label") or f"{pk.get('season') or ''} round-{pk.get('round') or '?'} pick".strip()
                    recv[to].append((None, str(label), 0))
            iso = str(t.get("iso") or "")
            out.append({"id": t.get("trade_id"), "season": sy, "iso": iso, "month": int(iso[5:7]) if iso[5:7].isdigit() else 0,
                        "teams": teams, "got": got, "recv": recv, "paper": dict(t.get("original_valuation") or {}),
                        "user": self.user in teams})
        out.sort(key=lambda x: (x["season"], x["iso"]))
        self._trades = out
        return out

    @staticmethod
    def net(tr: Dict[str, Any], tid: str) -> int:
        return tr["got"].get(tid, 0) - sum(v for k, v in tr["got"].items() if k != tid)

    def winner(self, tr: Dict[str, Any]) -> str:
        return max(tr["teams"], key=lambda t: self.net(tr, t))

    def moved(self, tr: Dict[str, Any]) -> str:
        bits = []
        for to, items in tr["recv"].items():
            if items:
                bits.append(f"{self.abbr(to)} got " + ", ".join(n for _, n, _ in items[:4]) + (" …" if len(items) > 4 else ""))
        return "; ".join(bits) or "no players recorded"

    def when(self, tr: Dict[str, Any]) -> str:
        return tr["iso"][:10] if tr["iso"] else f"{tr['season']}-{str(tr['season'] + 1)[2:]}"

    def on(self, tr: Dict[str, Any]) -> str:
        return f"on {tr['iso'][:10]}" if tr["iso"] else f"in {self.when(tr)}"

    def trade_rows(self, trs: List[Dict[str, Any]], tid: Optional[str] = None) -> Dict[str, Any]:
        rows = []
        for tr in trs[:10]:
            line = [self.when(tr), " / ".join(self.abbr(t) for t in tr["teams"]), self.moved(tr)]
            if tid:
                line.append(f"{self.net(tr, tid):+d}")
            else:
                w = self.winner(tr)
                line += [self.abbr(w), f"{self.net(tr, w):+d}"]
            rows.append(line)
        cols = ["When", "Teams", "What moved"] + ([f"Net for {self.abbr(tid)}"] if tid else ["Won by", "Margin"])
        return {"cols": cols, "rows": rows}

    # ---- drafts
    def drafts(self) -> Dict[str, Dict[str, Any]]:
        """pid -> draft year, overall pick, round, team, plus draft-day projections where kept."""
        if self._drafts is not None:
            return self._drafts
        rows: Dict[str, Dict[str, Any]] = {}
        for sy, s in self.seasons.items():
            for pk in s.get("draft") or []:
                pid = str(pk.get("player_id") or "")
                if pid and pk.get("overall"):
                    rows[pid] = {"year": int(pk.get("draft_year") or sy + 1), "overall": int(pk["overall"]),
                                 "round": int(pk.get("round") or 0), "team_id": str(pk.get("team_id") or ""), "in_save": True}
        for r in list(getattr(self.session, "draft_results_archive", None) or []):
            if not isinstance(r, dict):
                continue
            pid = str(r.get("prospect_id") or r.get("player_id") or "")
            ov = int(r.get("overall_pick") or r.get("overall") or 0)
            if not pid or not ov:
                continue
            row = rows.setdefault(pid, {"year": int(r.get("draft_year") or 0), "overall": ov, "round": int(r.get("round") or 0),
                                        "team_id": str(r.get("team_id") or ""), "in_save": True})
            for src, dst in (("drafted_potential", "draft_pot"), ("ceiling_grade", "draft_pot"), ("drafted_overall", "draft_ovr")):
                if r.get(src) and dst not in row:
                    try:
                        row[dst] = float(r[src])
                    except (TypeError, ValueError):
                        pass
        try:
            from services.award_history import _draft_payload_from_player
        except Exception:
            _draft_payload_from_player = None
        if _draft_payload_from_player is not None:
            for pid, p in self.view.players.items():
                if pid in rows:
                    continue
                try:
                    d = _draft_payload_from_player(p)
                except Exception:
                    continue
                if d.get("draft_overall_pick"):
                    ov = int(d["draft_overall_pick"])
                    rows[pid] = {"year": int(d.get("draft_year") or 0), "overall": ov, "round": (ov - 1) // 32 + 1, "in_save": False,
                                 "team_id": self.tid_of(d.get("drafted_by_team_id") or d.get("draft_team_id") or d.get("draft_team_abbr"))}
        # Generated players carry made-up draft slots; a slot claimed twice isn't a real draft record.
        slots = Counter((r["year"], r["overall"]) for r in rows.values() if not r.get("in_save"))
        rows = {pid: r for pid, r in rows.items() if r.get("in_save") or slots[(r["year"], r["overall"])] == 1}
        self._drafts = rows
        return rows

    def vs_slot(self, pid: str) -> Optional[float]:
        d = self.drafts().get(pid)
        now = self.ovr_now(pid)
        if not d or now is None:
            return None
        return round(now - draft_expectation(d["overall"]), 1)

    def drafted(self, pid: str) -> str:
        label = self.pick_label(pid)
        return label if label == "undrafted" else f"drafted {label}"

    def pick_label(self, pid: str) -> str:
        d = self.drafts().get(pid)
        if not d:
            return "undrafted"
        return f"#{d['overall']} in {d['year']}" if d.get("year") else f"#{d['overall']}"


def _hist(view: LiveView) -> Hist:
    h = view.__dict__.get("_hist")
    if h is None:
        h = Hist(view)
        view.__dict__["_hist"] = h
    return h


def _tbl(cols: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    return {"cols": cols, "rows": rows}


def _ord(n: Any) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _season(sy: int) -> str:
    return f"{sy}-{str(sy + 1)[2:]}"


def _no_history(h: Hist, what: str) -> Answer:
    return _answer(f"No {what} on record yet; this save has {h.span}.",
                   analysis=["It fills in as seasons finish; nothing here is invented."])


def _ours(raw: str) -> bool:
    return bool(re.search(r"\b(we|us|our|my|franchise|organi[sz]ation)\b", raw))


# =========================================================================== trades
def best_trade_for_us(h: Hist, raw: str) -> Answer:
    trs = [t for t in h.trades() if t["user"]]
    if not trs:
        return _answer("You haven't made a trade in this save yet, so there's no franchise trade to rank.",
                       analysis=[TRADE_RULE])
    trs.sort(key=lambda t: -h.net(t, h.user))
    t = trs[0]
    return _answer(f"Best trade in franchise history: {h.when(t)}, {h.moved(t)}. Net {h.net(t, h.user):+d} value for you.",
                   analysis=[TRADE_RULE, f"{len(trs)} trades on record since {min(h.seasons or [h.view.season])}."],
                   table=h.trade_rows(trs, h.user))


def worst_moves(h: Hist, raw: str, label: str = "Biggest mistake") -> Answer:
    trs = sorted([t for t in h.trades() if t["user"]], key=lambda t: h.net(t, h.user))
    busts = []
    for pid, d in h.drafts().items():
        if d.get("team_id") == h.user and d.get("round") == 1 and (h.view.row(pid).get("age") or 0) >= 21:
            gap = h.vs_slot(pid)
            if gap is not None:
                busts.append((gap, pid, d))
    busts.sort()
    analysis = [TRADE_RULE]
    if busts:
        gap, pid, d = busts[0]
        analysis.append(f"Worst pick: {h.name(pid)} ({h.pick_label(pid)}), {gap:+.0f} against his slot.")
    if trs and h.net(trs[0], h.user) < 0:
        t = trs[0]
        span = "" if "10" not in raw and "decade" not in raw else f" The save covers {h.span}, so a 10-year span isn't possible yet."
        return _answer(f"{label}: the {h.when(t)} trade, {h.moved(t)}. It has cost you {-h.net(t, h.user)} value so far.{span}",
                       analysis=analysis, table=h.trade_rows(trs, h.user))
    if busts:
        gap, pid, d = busts[0]
        return _answer(f"{label}: drafting {h.name(pid)} {h.pick_label(pid)}; he's {gap:+.0f} against what that slot usually becomes.",
                       analysis=[DRAFT_RULE, "None of your trades has lost value yet."])
    return _answer("No clear mistake on record: none of your trades has lost value and no first-round pick has fallen short.",
                   analysis=[TRADE_RULE, f"The save has {h.span}."])


def terrible_then_good(h: Hist, raw: str) -> Answer:
    trs = [t for t in h.trades() if t["user"] and t["paper"]]
    flips = []
    for t in trs:
        mine = float(t["paper"].get(h.user) or 0)
        theirs = max(float(v or 0) for k, v in t["paper"].items() if k != h.user) if len(t["paper"]) > 1 else mine
        if mine < theirs and h.net(t, h.user) > 0:
            flips.append((h.net(t, h.user) + (theirs - mine), t, theirs - mine))
    flips.sort(key=lambda x: -x[0])
    if not flips:
        return _answer("None of your trades fits: every deal that looked worse on paper has also played out worse, or you haven't lost one on paper.",
                       analysis=["“Looked terrible” uses the trade engine's valuation on the day; “helped” uses value since. " + TRADE_RULE])
    _, t, gap = flips[0]
    return _answer(f"The {h.when(t)} deal: {h.moved(t)}. On the day you gave up {gap:.0f} more trade value, but it's {h.net(t, h.user):+d} for you since.",
                   analysis=[TRADE_RULE], table=h.trade_rows([f[1] for f in flips], h.user))


def window_end(h: Hist, raw: str) -> Answer:
    cups = [sy for sy, t in h.champions() if t == h.user]
    finals = [sy for sy in h.seasons if h.finalist(sy) == h.user]
    good = sorted(set(cups + finals))
    if not good:
        return _answer("You haven't had a championship window in this save: no Cup or Final appearance yet.",
                       analysis=[f"The save has {h.span}."])
    last = max(good)
    if last == max(h.seasons):
        return _answer(f"Your window hasn't closed: you {'won the Cup' if last in cups else 'reached the Final'} in {_season(last)}, the latest finished season.")
    later = [t for t in h.trades() if t["user"] and t["season"] >= last]
    later.sort(key=lambda t: h.net(t, h.user))
    after = [h.pts(sy, h.user) for sy in sorted(h.seasons) if sy > last]
    note = f"Points after {_season(last)}: " + ", ".join(str(p) for p in after if p is not None) + "."
    if later and h.net(later[0], h.user) < 0:
        t = later[0]
        return _answer(f"The {h.when(t)} trade: {h.moved(t)}. It sent out the most value after your last run ({-h.net(t, h.user)} net).",
                       analysis=[note, TRADE_RULE], table=h.trade_rows(later, h.user))
    return _answer(f"No single move ended it: your last run was {_season(last)}, and no trade since has cost you value.",
                   analysis=[note, "The drop came from the roster aging or rivals improving, not a deal."])


def value_vs_cost(h: Hist, raw: str) -> Answer:
    rows = []
    for r in h.view.team_rows(h.user):
        cap = float((r.get("contract") or {}).get("cap_hit_m") or 0)
        car = h.career(r["id"])
        with_us = sum(h.val(r["id"], ln) for ln in h.lines(r["id"]) if str(ln.get("team_id")) == h.user)
        if cap <= 0 or with_us <= 0:
            continue
        rows.append((with_us / cap, r, with_us, cap))
    if not rows:
        return _answer("No one on your roster has production with you yet.")
    rows.sort(key=lambda x: -x[0])
    best = rows[0]
    return _answer(f"{best[1]['name']}: {best[2]} value for you on a ${best[3]:.2f}M cap hit, {best[0]:.0f} per $1M.",
                   analysis=["Cost is his current cap hit; value is everything he has produced in your uniform. " + VALUE],
                   table=_tbl(["Player", "Pos", "Value with you", "Cap hit", "Per $1M", "Acquired"],
                              [[r["name"], r["pos"], v, f"${c:.2f}M", f"{x:.0f}", h.pick_label(r["id"]) if h.drafts().get(r["id"], {}).get("team_id") == h.user else "trade/FA"]
                               for x, r, v, c in rows[:10]]), focus_ids=[best[1]["id"]])


def gave_up_early(h: Hist, raw: str, improved: bool = False) -> Answer:
    rows = []
    for pid in h.players:
        seasons = sorted(h.by_season(pid))
        mine = [ln for ln in h.lines(pid) if str(ln.get("team_id")) == h.user]
        if not mine:
            continue
        last_us = max(int(ln.get("season") or 0) for ln in mine)
        after = [ln for ln in h.lines(pid) if int(ln.get("season") or 0) >= last_us and str(ln.get("team_id") or "") not in ("", h.user)]
        if not after:
            continue
        gp_us, v_us = sum(int(l.get("gp") or 0) for l in mine), sum(h.val(pid, l) for l in mine)
        gp_af, v_af = sum(int(l.get("gp") or 0) for l in after), sum(h.val(pid, l) for l in after)
        age_now = h.view.row(pid).get("age") or 0
        age_left = age_now - (h.view.season - last_us) if age_now else None
        if improved:
            if gp_us < 10 or gp_af < 10:
                continue
            score = 82 * (v_af / gp_af - v_us / gp_us)
        else:
            if age_left and age_left > 26:
                continue
            score = v_af + (h.ovr_now(pid) or 0) / 10
        rows.append((score, pid, gp_us, v_us, gp_af, v_af, age_left, after[-1].get("team_id")))
    if not rows:
        return _answer("No former player of yours has gone on to produce elsewhere yet.",
                       analysis=[f"Needs players who left your club and played on; the save has {h.span}."])
    rows.sort(key=lambda x: -x[0])
    s, pid, gu, vu, ga, va, age, tid = rows[0]
    if improved:
        text = f"{h.name(pid)}: {vu} value in {gu} games with you, {va} in {ga} with the {h.team(tid)} ({s:+.0f} per 82 games)."
    else:
        text = f"{h.name(pid)}: left you at {age or '?'} and has {va} value in {ga} games since, now with the {h.team(tid)}."
    return _answer(text, analysis=[VALUE, "“Left” is his last season in your uniform; later lines are with other clubs."],
                   table=_tbl(["Player", "With you (GP / value)", "Since (GP / value)", "Age left", "Now"],
                              [[h.name(p), f"{a} / {b}", f"{c} / {d}", e or "—", h.abbr(f)] for _, p, a, b, c, d, e, f in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def robbery(h: Hist, raw: str) -> Answer:
    trs = sorted(h.trades(), key=lambda t: -abs(h.net(t, h.winner(t))))
    if not trs or h.net(trs[0], h.winner(trs[0])) <= 0:
        return _no_history(h, "lopsided trades")
    t = trs[0]
    w = h.winner(t)
    loser = next(x for x in t["teams"] if x != w)
    return _answer(f"The {h.when(t)} trade between the {h.team(w)} and {h.team(loser)}: {h.moved(t)}. The {h.abbr(w)} came out {h.net(t, w):+d} value ahead.",
                   analysis=[TRADE_RULE, f"{len(trs)} trades on record."], table=h.trade_rows(trs))


def balance_of_power(h: Hist, raw: str) -> Answer:
    best = None
    champs = dict(h.champions())
    for t in h.trades():
        w = h.winner(t)
        swing = h.net(t, w)
        if swing <= 0:
            continue
        before, after = h.rank(t["season"] - 1, w), h.rank(t["season"], w) or h.rank(t["season"] + 1, w)
        climb = (before - after) if before and after else 0
        cup = any(champs.get(sy) == w for sy in (t["season"], t["season"] + 1))
        score = swing + 2 * climb + (40 if cup else 0)
        if best is None or score > best[0]:
            best = (score, t, w, before, after, cup)
    if best is None:
        return _no_history(h, "trades with results")
    _, t, w, before, after, cup = best
    bits = [f"{h.net(t, w):+d} value for them"]
    if before and after:
        bits.append(f"league rank {before} → {after}")
    if cup:
        bits.append("and a Cup within a year")
    return _answer(f"The {h.when(t)} trade: {h.moved(t)}. It tilted things toward the {h.team(w)}: " + ", ".join(bits) + ".",
                   analysis=["Ranked by value swing, the winner's climb in the standings, and whether a Cup followed. " + TRADE_RULE])


def benefited_from_us(h: Hist, raw: str) -> Answer:
    tally: Dict[str, List[int]] = defaultdict(list)
    for t in h.trades():
        if not t["user"]:
            continue
        for x in t["teams"]:
            if x != h.user:
                tally[x].append(h.net(t, x))
    if not tally:
        return _answer("You haven't traded with anyone in this save yet.")
    rows = sorted(tally.items(), key=lambda kv: -sum(kv[1]))
    tid, nets = rows[0]
    if sum(nets) <= 0:
        return _answer("Nobody has come out ahead trading with you: every partner is level or behind.",
                       analysis=[TRADE_RULE], table=_tbl(["Team", "Trades", "Their net"], [[h.team(t), len(n), f"{sum(n):+d}"] for t, n in rows]))
    return _answer(f"The {h.team(tid)}: {len(nets)} trade{'s' if len(nets) != 1 else ''} with you, {sum(nets):+d} value in their favour.",
                   analysis=[TRADE_RULE], table=_tbl(["Team", "Trades", "Their net"], [[h.team(t), len(n), f"{sum(n):+d}"] for t, n in rows[:10]]))


def deadline_impact(h: Hist, raw: str) -> Answer:
    rows = []
    for t in h.trades():
        if t["month"] not in DEADLINE_MONTHS:
            continue
        for to, items in t["recv"].items():
            for pid, name, _ in items:
                if not pid:
                    continue
                po = [ln for ln in h.lines(pid, "playoffs") if int(ln.get("season") or 0) == t["season"] and str(ln.get("team_id")) == to]
                v = sum(h.val(pid, ln) for ln in po)
                gp = sum(int(ln.get("gp") or 0) for ln in po)
                rows.append((v, gp, name, to, t))
    if not rows:
        return _answer("No deadline acquisitions with playoff games on record.",
                       analysis=["Deadline means trades made in February or March; playoff lines are kept from seasons archived with playoffs."])
    rows.sort(key=lambda x: (-x[0], -x[1]))
    v, gp, name, to, t = rows[0]
    champ = dict(h.champions()).get(t["season"]) == to
    return _answer(f"{name}: picked up by the {h.team(to)} {h.on(t)}, then {v} playoff value in {gp} games{' and a Cup' if champ else ''}.",
                   analysis=[VALUE, "Deadline = trades made in February or March."],
                   table=_tbl(["Player", "Team", "Date", "Playoff GP", "Playoff value"], [[n, h.abbr(tt), h.when(tr), g, vv] for vv, g, n, tt, tr in rows[:10]]))


def traded_most(h: Hist, raw: str) -> Answer:
    c: Counter = Counter()
    names: Dict[str, str] = {}
    for t in h.trades():
        for items in t["recv"].values():
            for pid, name, _ in items:
                if pid:
                    c[pid] += 1
                    names[pid] = name
    if not c:
        return _no_history(h, "trades")
    rows = c.most_common(10)
    pid, n = rows[0]
    return _answer(f"{names[pid]}: traded {n} time{'s' if n != 1 else ''} in this save.",
                   analysis=[f"Counts trades since the save began ({h.span})."],
                   table=_tbl(["Player", "Times traded"], [[names[p], k] for p, k in rows]),
                   focus_ids=[pid] if pid in h.view.players else None)


def most_exchanged(h: Hist, raw: str) -> Answer:
    c: Counter = Counter()
    for t in h.trades():
        for items in t["recv"].values():
            for pid, name, _ in items:
                c[("player", name)] += 1
        for pk in (h.lg.get("trades") or {}).get(t["id"], {}).get("picks") or []:
            c[("pick", f"{h.abbr(pk.get('original_team_id'))} {pk.get('season') or ''} round {pk.get('round') or '?'}".strip())] += 1
    if not c:
        return _no_history(h, "trades")
    rows = c.most_common(10)
    (kind, label), n = rows[0]
    return _answer(f"{label}: moved {n} time{'s' if n != 1 else ''}, the most of any {kind}.",
                   table=_tbl(["Asset", "Kind", "Times moved"], [[l, k, n] for (k, l), n in rows]))


def superstar_moved(h: Hist, raw: str) -> Answer:
    best = None
    for t in h.trades():
        for to, items in t["recv"].items():
            for pid, name, _ in items:
                if not pid:
                    continue
                frm = next((x for x in t["teams"] if x != to), "")
                tenure = len({int(l.get("season") or 0) for l in h.lines(pid) if str(l.get("team_id")) == frm and int(l.get("season") or 0) < t["season"]})
                before = [l for l in h.lines(pid) if int(l.get("season") or 0) < t["season"] and l.get("ovr")]
                rating = float(before[-1]["ovr"]) if before else (h.ovr_now(pid) or 0)
                score = rating + 2 * tenure
                if best is None or score > best[0]:
                    best = (score, pid, name, frm, to, t, rating, tenure)
    if best is None:
        return _no_history(h, "trades")
    _, pid, name, frm, to, t, rating, tenure = best
    ten = f" after {tenure} season{'s' if tenure != 1 else ''} there" if tenure else ""
    return _answer(f"{name}: a {rating:.0f}-rated player the {h.team(frm)} moved to the {h.team(to)} {h.on(t)}{ten}.",
                   analysis=["“Unexpected” is read as the highest-rated player moved, with long tenure at his old club weighing extra."],
                   focus_ids=[pid] if pid in h.view.players else None)


def team_trade_record(h: Hist, raw: str) -> Answer:
    rec: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    for t in h.trades():
        for x in t["teams"]:
            n = h.net(t, x)
            rec[x][0 if n > 3 else 1 if n < -3 else 2] += 1
    if not rec:
        return _no_history(h, "trades")
    rows = sorted(rec.items(), key=lambda kv: (-(kv[1][0] - kv[1][1]), -kv[1][0]))
    tid, (w, l, e) = rows[0]
    return _answer(f"The {h.team(tid)}: {w} trades won, {l} lost, {e} even.",
                   analysis=["A trade is won when a side comes out more than 3 value ahead. " + TRADE_RULE],
                   table=_tbl(["Team", "Won", "Lost", "Even"], [[h.team(t), *v] for t, v in rows[:10]]))


def trade_to_cup(h: Hist, raw: str) -> Answer:
    rows = []
    for sy, champ in h.champions():
        for t in h.trades():
            if champ not in t["teams"] or not (sy - 2 <= t["season"] <= sy):
                continue
            v = 0
            names = []
            for pid, name, _ in t["recv"].get(champ, []):
                if not pid:
                    continue
                pv = sum(h.val(pid, l) for scope in ("regular", "playoffs") for l in h.lines(pid, scope)
                         if int(l.get("season") or 0) == sy and str(l.get("team_id")) == champ)
                if pv:
                    v += pv
                    names.append(f"{name} ({pv})")
            if v:
                rows.append((v, sy, champ, t, names))
    if not rows:
        return _answer("No Cup winner on record leaned on a trade: none of the champions got production from a player acquired in the two seasons before.",
                       analysis=[f"The save has {h.span}."])
    rows.sort(key=lambda x: -x[0])
    v, sy, champ, t, names = rows[0]
    hedge = " Without it they'd have lost that production; whether they'd still have won can't be replayed." if "would" in raw or "without" in raw else ""
    return _answer(f"The {h.when(t)} trade fed the {h.team(champ)}'s {_season(sy)} Cup: {', '.join(names)} in the Cup season.{hedge}",
                   analysis=["Ranked by what the incoming players produced for the champion that season, playoffs included. " + VALUE],
                   table=_tbl(["Cup", "Champion", "Trade", "Value in Cup season"], [[_season(s), h.abbr(c), h.when(tr), vv] for vv, s, c, tr, _ in rows[:10]]))


def counterfactual(h: Hist, raw: str) -> Answer:
    year = re.search(r"\b(19|20)\d{2}\b", raw)
    m = re.search(r"\bthe ([a-z'\-]+) (trade|deal)\b", raw)
    word = m.group(1) if m else ""
    trs = [t for t in h.trades() if (not word or any(word in n.lower() for items in t["recv"].values() for _, n, _ in items))]
    if year:
        y = int(year.group(0))
        trs = [t for t in trs if t["season"] in (y, y - 1) or t["iso"][:4] == str(y)]
    if not trs:
        what = f"a {word.title()} trade" if word else "that trade"
        when = f" in {year.group(0)}" if year else ""
        last = max(h.seasons) + 1 if h.seasons else h.view.season
        return _answer(f"There's no record of {what}{when}. This save's history runs to {last}" + (", so that season hasn't happened yet." if year and int(year.group(0)) > last else "."),
                       analysis=["Name a trade that's on record (“what if we never made the trade for X”) and I'll show what each side got from it."])
    t = max(trs, key=lambda x: abs(h.net(x, h.winner(x))))
    lines = []
    for tid in t["teams"]:
        lines.append(f"Without it, the {h.team(tid)} lose {t['got'][tid]} value they got from it ({h.net(t, tid):+d} net).")
    return _answer(f"The {h.when(t)} trade: {h.moved(t)}. " + lines[0],
                   analysis=lines[1:] + ["Games can't be replayed, so this is the production each side would not have had, not a re-simulated history."])


def butterfly(h: Hist, raw: str) -> Answer:
    trs = sorted(h.trades(), key=lambda t: -sum(t["got"].values()))
    if not trs:
        return _no_history(h, "trades")
    t = trs[0]
    return _answer(f"The {h.when(t)} trade: {h.moved(t)}. It moved {sum(t['got'].values())} value of production between clubs, the most of any deal.",
                   analysis=["Read as the move whose pieces went on to produce the most, on both sides. " + TRADE_RULE],
                   table=h.trade_rows(trs))


def single_event(h: Hist, raw: str) -> Answer:
    cands = []
    champs = dict(h.champions())
    for t in h.trades():
        w = h.winner(t)
        cup = any(champs.get(sy) == w for sy in (t["season"], t["season"] + 1))
        cands.append((h.net(t, w) + (60 if cup else 0), f"the {h.when(t)} trade ({h.moved(t)})" + (f", after which the {h.abbr(w)} won the Cup" if cup else "")))
    for sy, ser in h.series():
        win = str(ser["winner_id"])
        lose = str(ser["team_high_id"] if win == str(ser["team_low_id"]) else ser["team_low_id"])
        gap = (h.pts(sy, lose) or 0) - (h.pts(sy, win) or 0)
        if gap > 0:
            cup = champs.get(sy) == win
            cands.append((gap * 2 + (60 if cup else 0), f"the {h.team(win)} upsetting the {h.team(lose)} in {_season(sy)} ({gap} fewer points{', then winning the Cup' if cup else ''})"))
    if not cands:
        return _no_history(h, "trades or playoff series")
    cands.sort(key=lambda x: -x[0])
    return _answer(f"The biggest swing on record is {cands[0][1]}.",
                   analysis=["Compared trades by value swing and upsets by points gap, with extra weight when a Cup followed."] +
                            ([f"Runner-up: {cands[1][1]}."] if len(cands) > 1 else []))


def rejected_offer(h: Hist, raw: str) -> Answer:
    trs = sorted([t for t in h.trades() if t["user"]], key=lambda t: -h.net(t, h.user))
    return _answer("Offers you turned down aren't kept, so there's no way to grade them in hindsight.",
                   analysis=["What is kept is every trade you made; here they are graded by what each side got since. " + TRADE_RULE],
                   table=h.trade_rows(trs, h.user) if trs else None)


# =========================================================================== drafts
def _picks(h: Hist, team: Optional[str] = None, rounds: Optional[Tuple[int, int]] = None, active: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for pid, d in h.drafts().items():
        if team and d.get("team_id") != team:
            continue
        if rounds and not (rounds[0] <= (d.get("round") or 0) <= rounds[1]):
            continue
        if active and pid not in h.view.players:
            continue
        out.append((pid, d))
    return out


def _pick_rows(h: Hist, items: List[Tuple[Any, str]], extra: str = "Vs slot") -> Dict[str, Any]:
    rows = []
    for x, pid in items[:10]:
        d = h.drafts().get(pid, {})
        r = h.view.row(pid)
        rows.append([h.name(pid), r.get("pos") or "", h.abbr(d.get("team_id")), d.get("year") or "—", f"#{d.get('overall')}",
                     round(r.get("ovr") or 0), x if isinstance(x, str) else f"{x:+.0f}"])
    return _tbl(["Player", "Pos", "Drafted by", "Year", "Pick", "OVR", extra], rows)


def draft_steal(h: Hist, raw: str, team_only: bool = True, late: bool = False) -> Answer:
    team = h.user if team_only and _ours(raw) else None
    picks = _picks(h, team, (3, 99) if late else None)
    scored = sorted(((h.vs_slot(pid) or -99, pid) for pid, _ in picks), reverse=True) if not late else \
        sorted(((h.ovr_now(pid) or 0, pid) for pid, _ in picks), reverse=True)
    if not scored:
        who = "your club" if team else "anyone"
        return _answer(f"No drafted players of {who} are active yet to judge.", analysis=[DRAFT_RULE])
    x, pid = scored[0]
    if late:
        return _answer(f"{h.name(pid)}: a round-{h.drafts()[pid]['round']} pick ({h.pick_label(pid)}), now {x:.0f} overall, the best of anyone taken in round 3 or later.",
                       analysis=["Includes real-NHL draft history for players who were drafted before the save."],
                       table=_pick_rows(h, [(f"{v:.0f}", p) for v, p in scored], "OVR now"), focus_ids=[pid])
    whose = "Your biggest draft steal" if team else "Biggest draft steal"
    if x < 0:
        whose = "No pick has beaten its slot yet; the best value so far" if not team else "None of your picks has beaten its slot yet; the best so far"
    return _answer(f"{whose}: {h.name(pid)}, taken {h.pick_label(pid)}, now {h.ovr_now(pid):.0f} ({x:+.0f} against that slot).",
                   analysis=[DRAFT_RULE] + (["Includes players your club drafted before the save began."] if team else []),
                   table=_pick_rows(h, scored), focus_ids=[pid])


def first_round_bust(h: Hist, raw: str) -> Answer:
    team = h.user if _ours(raw) else None
    picks = [(pid, d) for pid, d in _picks(h, team, (1, 1)) if (h.view.row(pid).get("age") or 0) >= 21]
    if not picks:
        return _answer("No first-round pick of yours has had time to bust yet (21 or older)." if team else "No first-round picks old enough to judge.",
                       analysis=[DRAFT_RULE])
    scored = sorted((h.vs_slot(pid) or 0, pid) for pid, _ in picks)
    x, pid = scored[0]
    return _answer(f"{'Your biggest' if team else 'Biggest'} first-round bust: {h.name(pid)}, taken {h.pick_label(pid)}, now {h.ovr_now(pid):.0f} ({x:+.0f} against that slot).",
                   analysis=[DRAFT_RULE, "Only players 21 or older, so late bloomers aren't called busts early."],
                   table=_pick_rows(h, scored), focus_ids=[pid])


def pick_regret(h: Hist, raw: str) -> Answer:
    by_year: Dict[int, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for pid, d in h.drafts().items():
        if pid in h.view.players and d.get("year"):
            by_year[d["year"]].append((pid, d))
    rows = []
    for year, picks in by_year.items():
        for pid, d in picks:
            if d.get("team_id") != h.user:
                continue
            later = [(h.ovr_now(p) or 0, p) for p, dd in picks if dd["overall"] > d["overall"] and dd["overall"] <= d["overall"] + 32]
            if not later:
                continue
            best_ovr, best = max(later)
            gap = best_ovr - (h.ovr_now(pid) or 0)
            if gap > 0:
                rows.append((gap, pid, best, year))
    if not rows:
        return _answer("None of your picks looks costly yet: nobody taken in the next 32 picks has passed the player you chose.",
                       analysis=["Compares current ratings of active players only."])
    rows.sort(key=lambda x: -x[0])
    gap, pid, best, year = rows[0]
    db = h.drafts()[best]
    return _answer(f"Taking {h.name(pid)} {h.pick_label(pid)}: {h.name(best)} went #{db['overall']} to the {h.team(db.get('team_id'))} and is {gap:.0f} overall better today.",
                   analysis=["Looks at the next 32 picks after each of yours and compares current ratings."],
                   table=_tbl(["Year", "Your pick", "Pick", "Taken after", "Pick", "OVR gap"],
                              [[y, h.name(p), f"#{h.drafts()[p]['overall']}", h.name(b), f"#{h.drafts()[b]['overall']}", f"{g:+.0f}"] for g, p, b, y in rows[:10]]),
                   focus_ids=[best])


def _classes(h: Hist) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    for pid, d in h.drafts().items():
        if d.get("year"):
            out[int(d["year"])].append(pid)
    return out


def class_totals(h: Hist, raw: str, stat: str) -> Answer:
    rows = []
    for year, pids in _classes(h).items():
        gp = pts = 0
        for pid in pids:
            c, imp = h.career(pid), h.nhl_imported(pid)
            gp += c["gp"] + imp["gp"]
            pts += c["pts"] + imp["pts"]
        rows.append((gp if stat == "gp" else pts, year, len(pids), gp, pts))
    if not rows:
        return _no_history(h, "draft classes")
    rows.sort(key=lambda x: -x[0])
    _, year, n, gp, pts = rows[0]
    what = f"{gp} NHL games" if stat == "gp" else f"{pts} NHL points"
    return _answer(f"The {year} draft class: {what} from {n} players on record.",
                   analysis=["Counts NHL games and points before the save (where imported) plus every season in it, for drafted players still on record."],
                   table=_tbl(["Class", "Players", "NHL GP", "NHL points"], [[y, n, g, p] for _, y, n, g, p in rows[:10]]))


def strongest_class(h: Hist, raw: str) -> Answer:
    rows = []
    for year, pids in _classes(h).items():
        ovrs = sorted((h.ovr_now(p) or 0 for p in pids), reverse=True)[:10]
        if len(ovrs) < 5:
            continue
        stars = sum(1 for o in ovrs if o >= 90)
        rows.append((statistics.mean(ovrs), year, stars, len(pids)))
    if not rows:
        return _no_history(h, "draft classes")
    rows.sort(key=lambda x: -x[0])
    avg, year, stars, n = rows[0]
    return _answer(f"The {year} class: its best ten players average {avg:.1f} overall today, with {stars} at 90 or higher.",
                   analysis=["Judged on the top ten active players of each class, so depth and stars both count."],
                   table=_tbl(["Class", "Top-10 avg OVR", "90+ players", "Active"], [[y, f"{a:.1f}", s, k] for a, y, s, k in rows[:10]]))


def class_awards(h: Hist, raw: str) -> Answer:
    rows = []
    for year, pids in _classes(h).items():
        winners = [p for p in pids if h.awards_of(p)]
        if winners:
            rows.append((len(winners), year, winners))
    if not rows:
        return _answer("No drafted player has won a major award in this save yet, so no class leads.",
                       analysis=[f"Awards are counted from this save's ceremonies ({h.span})."])
    rows.sort(key=lambda x: -x[0])
    n, year, winners = rows[0]
    return _answer(f"The {year} class: {n} award winner{'s' if n != 1 else ''} ({', '.join(h.name(p) for p in winners[:4])}).",
                   table=_tbl(["Class", "Award winners", "Who"], [[y, k, ", ".join(h.name(p) for p in w[:4])] for k, y, w in rows[:10]]))


def gap_draft_success(h: Hist, raw: str) -> Answer:
    scored = sorted(((h.vs_slot(pid) or 0, pid) for pid, _ in _picks(h)), reverse=True)
    if not scored:
        return _no_history(h, "drafted players")
    x, pid = scored[0]
    low = min(scored)
    return _answer(f"{h.name(pid)}: taken {h.pick_label(pid)} and now {h.ovr_now(pid):.0f}, {x:+.0f} above what that slot usually becomes.",
                   analysis=[DRAFT_RULE, f"The other way: {h.name(low[1])} ({h.pick_label(low[1])}) is {low[0]:+.0f}."],
                   table=_pick_rows(h, scored), focus_ids=[pid])


def scouting_beat(h: Hist, raw: str, failed: bool = False) -> Answer:
    rows = []
    for pid, d in _picks(h):
        r = h.view.row(pid)
        proj = d.get("draft_pot")
        if proj:
            now = max(r.get("pot") or 0, r.get("ovr") or 0)
            rows.append((now - proj, pid, f"{proj:.0f} → {now:.0f}"))
        elif not failed:
            x = h.vs_slot(pid)
            if x is not None:
                rows.append((x, pid, f"slot {draft_expectation(d['overall']):.0f} → {r.get('ovr', 0):.0f}"))
    if failed:
        rows = [x for x in rows if (h.view.row(x[1]).get("age") or 0) >= 21]
        if not rows:
            # Without draft-day projections, the most-hyped are the top picks.
            rows = [((h.vs_slot(pid) or 0) - (6 if d["overall"] <= 5 else 0), pid, f"#{d['overall']} pick → {h.ovr_now(pid):.0f}")
                    for pid, d in _picks(h, rounds=(1, 1)) if (h.view.row(pid).get("age") or 0) >= 21 and (h.vs_slot(pid) or 0) <= -6]
        rows.sort()
    else:
        rows.sort(key=lambda x: -x[0])
    if not rows:
        if failed:
            return _answer("No highly touted pick has failed yet: every top-five pick or high draft-day projection is either still young or on track.",
                           analysis=["Only players 21 or older are judged."])
        return _no_history(h, "drafted prospects with scouting projections")
    x, pid, how = rows[0]
    if failed:
        text = f"{h.name(pid)} ({h.pick_label(pid)}): projected {how}, {x:+.0f} against what scouts saw on draft day."
    else:
        text = f"{h.name(pid)} ({h.pick_label(pid)}): {how}, {x:+.0f} past what scouts projected."
    return _answer(text, analysis=["Draft-day projection is the potential the scouts gave him at the draft; where it wasn't kept, his pick slot stands in."],
                   table=_tbl(["Player", "Drafted", "Projection → now", "Diff"], [[h.name(p), h.pick_label(p), w, f"{v:+.0f}"] for v, p, w in rows[:10]]),
                   focus_ids=[pid])


def bust_standings(h: Hist, raw: str) -> Answer:
    rows = []
    for pid, d in _picks(h, rounds=(1, 1)):
        if d["overall"] > 10 or (h.view.row(pid).get("age") or 0) < 21:
            continue
        x = h.vs_slot(pid)
        tid = d.get("team_id")
        ranks = [h.rank(sy, tid) for sy in sorted(h.seasons) if sy >= d.get("year", 0)]
        ranks = [r for r in ranks if r]
        rows.append(((x or 0) - (statistics.mean(ranks) / 4 if ranks else 0), pid, x, tid, ranks))
    if not rows:
        return _answer("No top-ten pick has busted yet (21 or older and well below his slot).", analysis=[DRAFT_RULE])
    rows.sort()
    _, pid, x, tid, ranks = rows[0]
    tail = f" The {h.team(tid)} have finished " + ", ".join(_ord(r) for r in ranks) + " in the league since." if ranks and tid else ""
    whose = f" by the {h.team(tid)}" if tid else ""
    return _answer(f"{h.name(pid)}, picked {h.pick_label(pid)}{whose}: {x:+.0f} against his slot.{tail}",
                   analysis=[DRAFT_RULE, "Weighs how far below his slot he is with how low his team has finished since."], focus_ids=[pid])


def lottery(h: Hist, raw: str) -> Answer:
    firsts = sorted(((d["year"], pid) for pid, d in h.drafts().items() if d["overall"] == 1 and d.get("year")), reverse=True)
    if not firsts:
        return _answer("Lottery results aren't archived, and no first-overall picks are on record.")
    rows = []
    for year, pid in firsts[:10]:
        second = next((p for p, d in h.drafts().items() if d.get("year") == year and d["overall"] == 2), None)
        rows.append([year, h.name(pid), h.abbr(h.drafts()[pid].get("team_id")), round(h.ovr_now(pid) or 0) or "—",
                     h.name(second) if second else "—", round(h.ovr_now(second) or 0) if second else "—"])
    best = max(firsts, key=lambda yp: h.ovr_now(yp[1]) or 0)
    return _answer(f"Lottery draws aren't archived, but the #1 pick is what a lottery decides: the best one on record is {h.name(best[1])} ({best[0]}{', ' + h.team(h.drafts()[best[1]].get('team_id')) if h.drafts()[best[1]].get('team_id') else ''}), now {h.ovr_now(best[1]) or 0:.0f}.",
                   table=_tbl(["Year", "#1 pick", "Team", "OVR", "#2 pick", "OVR"], rows))


def org_benefit(h: Hist, raw: str) -> Answer:
    rows = []
    only = h.user if _ours(raw) else None
    for pid, d in h.drafts().items():
        tid = d.get("team_id")
        if not tid or (only and tid != only):
            continue
        with_team = [l for l in h.lines(pid) if str(l.get("team_id")) == tid]
        if not with_team:
            continue
        v = sum(h.val(pid, l) for l in with_team)
        cups = [c for c in h.cups_of(pid) if dict(h.champions()).get(c) == tid]
        rows.append((v + 50 * len(cups), pid, tid, v, len(cups)))
    if not rows:
        return _no_history(h, "drafted players who played for their drafting club")
    rows.sort(key=lambda x: -x[0])
    _, pid, tid, v, cups = rows[0]
    return _answer(f"{h.name(pid)}: drafted by the {h.team(tid)} ({h.pick_label(pid)}) and has {v} value for them" + (f" and {cups} Cup{'s' if cups != 1 else ''}." if cups else "."),
                   analysis=["Value produced for the club that drafted him, with each Cup won there counted as 50. " + VALUE],
                   table=_tbl(["Player", "Drafted by", "Pick", "Value there", "Cups there"], [[h.name(p), h.abbr(t), h.pick_label(p), vv, c] for _, p, t, vv, c in rows[:10]]))


# =========================================================================== playoffs
def _loser(ser: Dict[str, Any]) -> str:
    w = str(ser["winner_id"])
    return str(ser["team_high_id"]) if w == str(ser["team_low_id"]) else str(ser["team_low_id"])


def _games(ser: Dict[str, Any]) -> int:
    return int(ser.get("wins_high") or 0) + int(ser.get("wins_low") or 0)


def _need_playoffs(h: Hist) -> Optional[Answer]:
    if h.series():
        return None
    return _answer(f"No playoff series are archived yet; this save has {h.span}" +
                   (" and those seasons were rebuilt without brackets." if h.seasons else "."),
                   analysis=["Series are kept from every playoff played from here on."])


# ---- game-by-game playoff results (kept for playoffs archived with box scores)
NO_GAMES = "Game-by-game playoff results are kept for every playoff archived from now on; none of the archived series has them yet."


def _po_games(h: Hist) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    return [(sy, ser, g) for sy, ser in h.series() for g in ser.get("games") or []]


def _blown(ser: Dict[str, Any]) -> int:
    """Largest series lead the eventual loser held."""
    w, l = str(ser["winner_id"]), _loser(ser)
    wins = {w: 0, l: 0}
    best = 0
    for g in ser.get("games") or []:
        if g.get("winner_id") in wins:
            wins[g["winner_id"]] += 1
            best = max(best, wins[l] - wins[w])
    return best


def game7_goals(h: Hist, raw: str) -> Answer:
    games = [(sy, ser, g) for sy, ser, g in _po_games(h) if int(g.get("game") or 0) == 7]
    if not _po_games(h):
        sevens = sum(1 for _, ser in h.series() if _games(ser) == 7)
        return _answer(NO_GAMES, analysis=[f"{sevens} archived series went to a seventh game." if h.series() else f"The save has {h.span}."])
    c: Counter = Counter()
    for sy, ser, g in games:
        for _, pid, _ in g.get("goals") or []:
            c[pid] += 1
    if not c:
        return _answer("No Game 7 goals on record: no archived series has gone seven games.")
    rows = c.most_common(10)
    pid, n = rows[0]
    return _answer(f"{h.name(pid)}: {n} Game 7 goal{'s' if n != 1 else ''}.", analysis=[f"From {len(games)} archived Game 7s."],
                   table=_tbl(["Player", "Game 7 goals"], [[h.name(p), k] for p, k in rows]), focus_ids=[pid] if pid in h.view.players else None)


def biggest_goal(h: Hist, raw: str) -> Answer:
    team = h.user if _ours(raw) else None
    rows = []
    for sy, ser in h.series():
        games = ser.get("games") or []
        if not games:
            continue
        last = games[-1]
        w = str(ser["winner_id"])
        if team and w != team:
            continue
        final = int(ser.get("round_index") or 0) == h.final_round(sy)
        gwg = last.get("gwg")
        if not gwg:
            goals = [x for x in last.get("goals") or [] if x[0] == w]
            gwg = goals[-1][1] if goals else None
        if not gwg:
            continue
        weight = 10 * int(ser.get("round_index") or 0) + (40 if final else 0) + (15 if int(last.get("game") or 0) == 7 else 0) + (10 if last.get("ot_periods") else 0)
        what = ("Cup-winning" if final else f"series-clinching (round {ser.get('round_index')})") + (" overtime" if last.get("ot_periods") else "") + \
            (" Game 7" if int(last.get("game") or 0) == 7 else f" Game {last.get('game')}")
        rows.append((weight, gwg, sy, w, _loser(ser), what))
    if not rows:
        if not _po_games(h):
            return _answer(NO_GAMES, analysis=["The biggest goal is read as the winner in the deepest clinching game, so it needs those results."])
        return _answer("Your club hasn't won a playoff series with game results on record yet." if team else "No clinching goals on record yet.")
    rows.sort(key=lambda x: -x[0])
    _, pid, sy, w, l, what = rows[0]
    return _answer(f"{h.name(pid)}: the {what} winner for the {h.team(w)} against the {h.team(l)} in {_season(sy)}.",
                   analysis=["Biggest = game-winner in the deepest clinching game, with extra weight for a Cup, Game 7 or overtime."],
                   table=_tbl(["Scorer", "Season", "Team", "Goal"], [[h.name(p), _season(s), h.abbr(t), wh] for _, p, s, t, _, wh in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def blown_lead(h: Hist, raw: str) -> Answer:
    if not _po_games(h):
        return heartbreak(h, raw, blew_lead=True)
    rows = [(_blown(ser), int(ser.get("round_index") or 0), sy, ser) for sy, ser in h.series() if ser.get("games")]
    rows = [r for r in rows if r[0] >= 2]
    if not rows:
        return _answer("No team has blown a two-game series lead in the archived playoffs.")
    rows.sort(key=lambda x: (-x[0], -x[1]))
    lead, rnd, sy, ser = rows[0]
    l, w = _loser(ser), str(ser["winner_id"])
    return _answer(f"The {_season(sy)} {h.team(l)}: up {lead + 0} games in round {rnd}, then lost to the {h.team(w)} in {_games(ser)}.",
                   analysis=["Lead = most games the eventual loser was ahead at any point in the series."],
                   table=_tbl(["Season", "Blew it", "To", "Round", "Biggest lead", "Games"], [[_season(s), h.abbr(_loser(x)), h.abbr(x["winner_id"]), r, ld, _games(x)] for ld, r, s, x in rows[:10]]))


def upset(h: Hist, raw: str) -> Answer:
    miss = _need_playoffs(h)
    if miss:
        return miss
    rows = []
    for sy, ser in h.series():
        w, l = str(ser["winner_id"]), _loser(ser)
        gap = (h.pts(sy, l) or 0) - (h.pts(sy, w) or 0)
        if gap > 0:
            rows.append((gap, sy, w, l, ser))
    if not rows:
        return _answer("No upsets on record: the team with more points has won every series.")
    rows.sort(key=lambda x: -x[0])
    gap, sy, w, l, ser = rows[0]
    rnd = "the Final" if int(ser.get("round_index") or 0) == h.final_round(sy) else f"round {ser.get('round_index')}"
    return _answer(f"{_season(sy)}, {rnd}: the {h.team(w)} beat the {h.team(l)}, who had {gap} more points, in {_games(ser)} games.",
                   analysis=["Upset size is the regular-season points gap."],
                   table=_tbl(["Season", "Round", "Winner", "Loser", "Points gap", "Games"],
                              [[_season(s), x.get("round_index"), h.abbr(a), h.abbr(b), g, _games(x)] for g, s, a, b, x in rows[:10]]))


def weakest_champion(h: Hist, raw: str) -> Answer:
    rows = [(h.pts(sy, t) or 0, sy, t, h.rank(sy, t)) for sy, t in h.champions()]
    if not rows:
        return _no_history(h, "Cup winners")
    rows.sort()
    pts, sy, t, rank = rows[0]
    return _answer(f"The {_season(sy)} {h.team(t)}: {pts} points, {_ord(rank)} in the league, the fewest of any champion.",
                   table=_tbl(["Season", "Champion", "Points", "League rank"], [[_season(s), h.team(c), p, r] for p, s, c, r in rows]))


def improbable_run(h: Hist, raw: str, any_team: bool = False) -> Answer:
    miss = _need_playoffs(h)
    if miss and not h.champions():
        return miss
    rows = []
    for sy in h.seasons:
        teams = {str(t) for ser in h.seasons[sy].get("playoff_series") or [] for t in (ser.get("team_high_id"), ser.get("team_low_id"))}
        if not teams and h.seasons[sy].get("champion_id"):
            teams = {str(h.seasons[sy]["champion_id"])}
        for t in teams:
            reached = max([int(s.get("round_index") or 0) for s in h.seasons[sy].get("playoff_series") or [] if t in (str(s.get("team_high_id")), str(s.get("team_low_id")))] or [0])
            won = str(h.seasons[sy].get("champion_id")) == t
            seed = sorted(teams, key=lambda x: -(h.pts(sy, x) or 0)).index(t) + 1
            expected = 1 + (len(teams) - seed) / max(1, len(teams)) * max(1, h.final_round(sy) - 1)
            score = (reached + (1 if won else 0)) - expected
            rows.append((score, sy, t, reached, won, h.rank(sy, t), seed))
    if not any_team:
        rows = [r for r in rows if r[4] or h.finalist(r[1]) == r[2]] or rows
    if not rows:
        return _no_history(h, "playoff runs")
    rows.sort(key=lambda x: -x[0])
    _, sy, t, reached, won, rank, seed = rows[0]
    result = "won the Cup" if won else ("reached the Final" if reached == h.final_round(sy) else f"reached round {reached}")
    return _answer(f"The {_season(sy)} {h.team(t)}: {_ord(rank)} in the league in the regular season, seeded {_ord(seed)} of the playoff teams, and {result}.",
                   analysis=["Measured against how far a team with that seed usually goes."],
                   table=_tbl(["Season", "Team", "League rank", "Playoff seed", "Result"],
                              [[_season(s), h.abbr(x), r, sd, "Cup" if w else f"round {rc}"] for _, s, x, rc, w, r, sd in rows[:10]]))


def goalie_carry(h: Hist, raw: str) -> Answer:
    rows = []
    for pid, pr in h.players.items():
        if not str(pr.get("position") or "").upper().startswith("G"):
            continue
        for ln in h.lines(pid, "playoffs"):
            w = int(ln.get("w") or 0)
            if not w:
                continue
            sy, tid = int(ln.get("season") or 0), str(ln.get("team_id") or "")
            rank = h.rank(sy, tid) or 16
            rows.append((w + rank / 8, pid, sy, tid, w, ln.get("sv_pct"), rank))
    if not rows:
        miss = _need_playoffs(h)
        return miss or _answer("No goalie playoff lines on record yet.")
    rows.sort(key=lambda x: -x[0])
    _, pid, sy, tid, w, sv, rank = rows[0]
    sv_txt = f", {float(sv):.3f} save percentage" if sv else ""
    return _answer(f"{h.name(pid)}, {_season(sy)} {h.team(tid)}: {w} playoff wins{sv_txt}, for a team that was {_ord(rank)} in the regular season.",
                   analysis=["Playoff wins, with extra credit the weaker his team was in the regular season."],
                   table=_tbl(["Goalie", "Season", "Team", "Playoff W", "SV%", "Team rank"],
                              [[h.name(p), _season(s), h.abbr(t), ww, f"{float(v):.3f}" if v else "—", r] for _, p, s, t, ww, v, r in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def closest_final(h: Hist, raw: str) -> Answer:
    rows = []
    for sy, ser in h.series():
        if int(ser.get("round_index") or 0) == h.final_round(sy):
            rows.append((_games(ser), sy, ser))
    if not rows:
        return _need_playoffs(h) or _no_history(h, "Cup Finals")
    def diff(ser: Dict[str, Any]) -> int:
        w = str(ser["winner_id"])
        return sum((g["home_score"] - g["away_score"]) * (1 if g["home_id"] == w else -1) for g in ser.get("games") or [])

    rows.sort(key=lambda x: (-x[0], diff(x[2]) if x[2].get("games") else 99))
    g, sy, ser = rows[0]
    w, l = str(ser["winner_id"]), _loser(ser)
    margin = f", outscoring them by {diff(ser)} over the series" if ser.get("games") else ""
    return _answer(f"{_season(sy)}: the {h.team(w)} beat the {h.team(l)} in {g} games{margin}.",
                   analysis=["Closeness is the number of games, then the series goal difference where game scores are archived."],
                   table=_tbl(["Season", "Champion", "Finalist", "Games"], [[_season(s), h.abbr(x["winner_id"]), h.abbr(_loser(x)), gg] for gg, s, x in rows]))


def heartbreak(h: Hist, raw: str, blew_lead: bool = False) -> Answer:
    miss = _need_playoffs(h)
    if miss:
        return miss
    rows = []
    for sy, ser in h.series():
        l = _loser(ser)
        final = int(ser.get("round_index") or 0) == h.final_round(sy)
        g7 = _games(ser) == 7
        pain = (h.pts(sy, l) or 0) / 4 + (25 if final else 0) + (20 if g7 else 0) + 4 * int(ser.get("round_index") or 0)
        if blew_lead and not g7:
            continue
        rows.append((pain, sy, l, str(ser["winner_id"]), ser, final, g7))
    if not rows:
        return _answer("No series has gone the full seven games yet, so no lead has been blown that far.",
                       analysis=["Game-by-game order inside past series isn't archived; a seven-game loss is the nearest record of a blown lead."])
    rows.sort(key=lambda x: -x[0])
    _, sy, l, w, ser, final, g7 = rows[0]
    what = "the Final" if final else f"round {ser.get('round_index')}"
    lead = " Game-by-game order isn't archived, so this is the seven-game loss that hurt most, not a confirmed blown lead." if blew_lead else ""
    return _answer(f"The {_season(sy)} {h.team(l)}: {h.pts(sy, l)} points in the regular season, then lost {what} to the {h.team(w)}{' in seven games' if g7 else ''}.{lead}",
                   analysis=["Weighs how good the losing team was, how deep the series was, and whether it went seven."],
                   table=_tbl(["Season", "Eliminated", "By", "Round", "Games"], [[_season(s), h.abbr(a), h.abbr(b), x.get("round_index"), _games(x)] for _, s, a, b, x, _, _ in rows[:10]]))


def _count(n: int, what: str) -> str:
    if n == 1:
        return f"1 {what}"
    head, _, tail = what.partition(" lost")
    head = head + "es" if head.endswith("loss") else head + "s"
    return f"{n} {head}{' lost' if tail or what.endswith(' lost') else ''}"


def most_what_ifs(h: Hist, raw: str) -> Answer:
    tally: Dict[str, Counter] = defaultdict(Counter)
    for sy, ser in h.series():
        l = _loser(ser)
        if _games(ser) == 7:
            tally[l]["game-7 loss"] += 1
        if int(ser.get("round_index") or 0) == h.final_round(sy):
            tally[l]["lost Final"] += 1
    for t in h.trades():
        for x in t["teams"]:
            if h.net(t, x) < -20:
                tally[x]["lopsided trade lost"] += 1
    for pid, d in _picks(h, rounds=(1, 1)):
        if d["overall"] <= 10 and (h.vs_slot(pid) or 0) < -8:
            tally[d.get("team_id") or ""]["top-10 bust"] += 1
    tally.pop("", None)
    if not tally:
        return _no_history(h, "near-misses")
    rows = sorted(tally.items(), key=lambda kv: -sum(kv[1].values()))
    tid, c = rows[0]
    return _answer(f"The {h.team(tid)}: " + ", ".join(_count(n, k) for k, n in c.most_common()) + ".",
                   analysis=["Counts game-7 losses, lost Finals, trades lost by 20+ value and top-10 picks well below their slot."],
                   table=_tbl(["Team", "What-ifs", "Detail"], [[h.team(t), sum(cc.values()), ", ".join(_count(n, k) for k, n in cc.most_common())] for t, cc in rows[:10]]))


def dynasty_close(h: Hist, raw: str) -> Answer:
    cups = Counter(t for _, t in h.champions())
    dyn = [t for t, n in cups.items() if n >= 2]
    if not dyn:
        return _answer("No dynasty yet: no club has won more than one Cup in this save.", analysis=[f"The save has {h.span}."])
    if not h.series():
        return fb_dynasty(h, raw)
    rows = []
    for t in dyn:
        close = [(sy, ser) for sy, ser in h.series() if str(ser["winner_id"]) == t and _games(ser) == 7]
        rows.append((len(close), t, cups[t], close))
    rows.sort(key=lambda x: -x[0])
    n, t, k, close = rows[0]
    if not close:
        return _answer(f"The {h.team(t)}' {k} Cups: never closer than six games in any series they won." if h.series() else
                       f"The {h.team(t)} have {k} Cups, but their series weren't archived, so how close each run came isn't known.")
    sy, ser = close[0]
    return _answer(f"The {h.team(t)} ({k} Cups): {n} of their series wins went seven games, the first in {_season(sy)} against the {h.team(_loser(ser))}.",
                   analysis=["Closest to never happening = most series won in a seventh game."])


def dynasty_start(h: Hist, raw: str) -> Answer:
    cups = sorted(sy for sy, t in h.champions() if t == h.user)
    if not cups:
        best = max(h.seasons, key=lambda sy: h.pts(sy, h.user) or 0) if h.seasons else None
        return _answer("You haven't won a Cup in this save, so there's no dynasty yet." +
                       (f" Your best season was {_season(best)} ({h.pts(best, h.user)} points)." if best else ""))
    start = cups[0]
    prev = h.rank(start - 1, h.user)
    tail = f" The season before, you finished {_ord(prev)} in the league." if prev else " It was the save's first season."
    return _answer(f"{_season(start)}: your first Cup in this save, the start of {len(cups)} in {cups[-1] - start + 1} seasons.{tail}",
                   table=_tbl(["Season", "Points", "League rank", "Cup"], [[_season(sy), h.pts(sy, h.user), h.rank(sy, h.user), "yes" if sy in cups else ""] for sy in sorted(h.seasons)]))


# =========================================================================== careers
def _peak(h: Hist, pid: str) -> float:
    path = h.ovr_path(pid)
    return max((o for _, o in path), default=h.ovr_now(pid) or 0)


def strangest_path(h: Hist, raw: str) -> Answer:
    rows = []
    trades = Counter(pid for t in h.trades() for items in t["recv"].values() for pid, _, _ in items if pid)
    for pid in h.players:
        c, imp = h.career(pid), h.nhl_imported(pid)
        teams = len(c["teams"]) + imp["teams"]
        if teams < 3:
            continue
        rows.append((teams + trades[pid], pid, teams, trades[pid], c["seasons"] + imp["seasons"]))
    if not rows:
        return _no_history(h, "multi-team careers")
    rows.sort(key=lambda x: -x[0])
    _, pid, teams, n, seasons = rows[0]
    path = " → ".join(h.abbr(t) for t in h.career(pid)["teams"])
    return _answer(f"{h.name(pid)}: {teams} NHL clubs in {seasons} seasons, traded {n} time{'s' if n != 1 else ''} in this save ({path}).",
                   analysis=["Strangest = most clubs and most trades; pre-save clubs count where the real career was imported."],
                   table=_tbl(["Player", "Clubs", "Trades in save", "Seasons"], [[h.name(p), t, k, s] for _, p, t, k, s in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def declined_fastest(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        path = h.ovr_path(pid)
        if len(path) >= 2:
            peak_i = max(range(len(path)), key=lambda i: path[i][1])
            peak_y, peak = path[peak_i]
            if peak >= 88 and peak_i < len(path) - 1:
                low_y, low = path[-1]
                rows.append(((peak - low) / max(1, low_y - peak_y), pid, f"{peak:.0f} → {low:.0f} OVR", peak_y, low_y))
                continue
        bs = {y: st for y, st in h.by_season(pid).items() if st["gp"] >= 40}
        ys = sorted(bs)
        if len(ys) >= 2:
            rate = {y: 82 * bs[y]["val"] / bs[y]["gp"] for y in ys}
            peak_y = max(ys, key=lambda y: rate[y])
            if rate[peak_y] >= 70 and peak_y != ys[-1]:
                drop = rate[peak_y] - rate[ys[-1]]
                rows.append((drop / 10 / max(1, ys[-1] - peak_y), pid, f"{rate[peak_y]:.0f} → {rate[ys[-1]]:.0f} value per 82 games", peak_y, ys[-1]))
    if not rows:
        return _answer("No star has fallen off yet: nobody has dropped meaningfully from a peak season.", analysis=[f"The save has {h.span}."])
    rows.sort(key=lambda x: -x[0])
    _, pid, how, a, b = rows[0]
    return _answer(f"{h.name(pid)}: {how} between {a} and {b}.",
                   analysis=["Ratings at each season end where recorded; otherwise scoring rate (40+ games a season) from a 70+ peak."],
                   table=_tbl(["Player", "Drop", "Peak", "Latest"], [[h.name(p), w, x, y] for _, p, w, x, y in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def one_season_wonder(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        bs = {y: s for y, s in h.by_season(pid).items() if s["gp"] >= 20}
        if len(bs) < 3:
            continue
        vals = sorted((s["val"] for s in bs.values()), reverse=True)
        if vals[0] < 50:
            continue
        best_y = max(bs, key=lambda y: bs[y]["val"])
        rest = statistics.mean(vals[1:])
        rows.append((vals[0] - rest, pid, best_y, vals[0], rest))
    if not rows:
        return _answer("No one-season wonder yet: that needs players with three or more seasons and one standout year.",
                       analysis=[f"The save has {h.span}."])
    rows.sort(key=lambda x: -x[0])
    gap, pid, y, best, rest = rows[0]
    return _answer(f"{h.name(pid)}: {best} value in {_season(y)}, against an average of {rest:.0f} in his other seasons.",
                   analysis=[VALUE, "Seasons of 20+ games; best season must be 50+."],
                   table=_tbl(["Player", "Best season", "Value", "Other seasons avg"], [[h.name(p), _season(s), b, f"{r:.0f}"] for _, p, s, b, r in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def best_no_award(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        if h.awards_of(pid):
            continue
        c = h.career(pid)
        if c["val"] > 0:
            rows.append((c["val"], pid, c["gp"], c["seasons"]))
    if not rows:
        return _no_history(h, "careers")
    rows.sort(key=lambda x: -x[0])
    v, pid, gp, s = rows[0]
    return _answer(f"{h.name(pid)}: {v} value in {gp} games over {s} season{'s' if s != 1 else ''}, and no major award.",
                   analysis=[VALUE, "Awards counted are this save's major trophies; real-life awards before the save aren't on record."],
                   table=_tbl(["Player", "Value", "GP", "Seasons"], [[h.name(p), vv, g, ss] for vv, p, g, ss in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def longest_career(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        c, imp = h.career(pid), h.nhl_imported(pid)
        seasons, gp = c["seasons"] + imp["seasons"], c["gp"] + imp["gp"]
        if seasons:
            rows.append((seasons, gp, pid))
    if not rows:
        return _no_history(h, "careers")
    rows.sort(key=lambda x: (-x[0], -x[1]))
    s, gp, pid = rows[0]
    return _answer(f"{h.name(pid)}: {s} NHL seasons and {gp} games.",
                   analysis=["Real NHL seasons before the save count where they were imported, plus every season in it."],
                   table=_tbl(["Player", "Seasons", "GP", "Status"], [[h.name(p), ss, g, "retired" if (h.players.get(p) or {}).get("retired_season") else "active"] for ss, g, p in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def biggest_rise(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.view.players:
        now = h.ovr_now(pid)
        base = [o for _, o in h.ovr_path(pid)[:-1]]
        d = h.drafts().get(pid, {})
        if d.get("draft_ovr"):
            base.append(d["draft_ovr"])
        start = h.view.row(pid).get("season_start_ovr")
        if start:
            base.append(float(start))
        if now and base:
            low = min(base)
            rows.append((now - low, pid, low, now, "draft day" if d.get("draft_ovr") == low else "earliest record"))
    if not rows:
        return _no_history(h, "rating histories")
    rows.sort(key=lambda x: -x[0])
    gain, pid, low, now, src = rows[0]
    return _answer(f"{h.name(pid)}: {low:.0f} → {now:.0f} overall, +{gain:.0f} since his {src}.",
                   analysis=["Earliest rating on record: draft day for players drafted in the save, otherwise the first season-end rating kept."],
                   table=_tbl(["Player", "From", "Now", "Gain", "Since"], [[h.name(p), f"{a:.0f}", f"{b:.0f}", f"+{g:.0f}", s] for g, p, a, b, s in rows[:10]]),
                   focus_ids=[pid])


def breakout(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        bs = {y: s for y, s in h.by_season(pid).items() if s["gp"] >= 20}
        ys = sorted(bs)
        for a, b in zip(ys, ys[1:]):
            jump = bs[b]["val"] - bs[a]["val"]
            if jump > 0:
                rows.append((jump, pid, b, bs[a]["val"], bs[b]["val"]))
    if not rows:
        return _answer("No breakout on record yet: that needs two seasons of 20+ games to compare.", analysis=[f"The save has {h.span}."])
    rows.sort(key=lambda x: -x[0])
    j, pid, y, a, b = rows[0]
    return _answer(f"{h.name(pid)} in {_season(y)}: {a} → {b} value, +{j} on the season before.",
                   analysis=[VALUE, "Season-over-season jump, 20+ games in both."],
                   table=_tbl(["Player", "Season", "Before", "Breakout", "Jump"], [[h.name(p), _season(s), x, z, f"+{jj}"] for jj, p, s, x, z in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def underrated_career(h: Hist, raw: str) -> Answer:
    pts = []
    for pid in h.players:
        if h.goalie(pid) or pid not in h.view.players:
            continue
        c = h.career(pid)
        if c["gp"] >= 82:
            pts.append((pid, 82 * c["pts"] / c["gp"], h.ovr_now(pid) or 0))
    if len(pts) < 10:
        return _no_history(h, "careers long enough (82+ games)")
    mx, my = statistics.mean(p[2] for p in pts), statistics.mean(p[1] for p in pts)
    cov = sum((p[2] - mx) * (p[1] - my) for p in pts)
    var = sum((p[2] - mx) ** 2 for p in pts) or 1
    slope = cov / var
    rows = sorted(((p82 - (my + slope * (o - mx)), pid, p82, o) for pid, p82, o in pts if not h.awards_of(pid)), reverse=True)
    res, pid, p82, o = rows[0]
    return _answer(f"{h.name(pid)}: {p82:.0f} points per 82 games in this save, {res:+.0f} more than players rated {o:.0f} usually score, and no major award.",
                   analysis=["Career scoring rate against what his rating predicts, skaters with 82+ games, award winners excluded."],
                   table=_tbl(["Player", "OVR", "Pts/82", "Above rating"], [[h.name(p), f"{oo:.0f}", f"{x:.0f}", f"{r:+.0f}"] for r, p, x, oo in rows[:10]]),
                   focus_ids=[pid])


def retired_prime(h: Hist, raw: str) -> Answer:
    from .answers3 import _weakest_group

    retired = h.retired()
    if not retired:
        return _answer("Nobody has retired in this save yet, so there's no one to bring back.")
    grp, _ = _weakest_group(h.view, h.user)
    want = {"G": "G", "D": "D", "C": "C", "W": "W"}.get(grp, "")

    def fits(pid: str) -> bool:
        pos = str((h.players.get(pid) or {}).get("position") or "").upper()
        return not want or (want == "W" and pos in ("LW", "RW", "W")) or pos.startswith(want)

    rows = []
    for pid in retired:
        c = h.career(pid)
        ret = (h.players.get(pid) or {}).get("retirement") or {}
        peak = max([float(l["ovr"]) for l in h.lines(pid) if l.get("ovr")] + ([float(ret["overall"])] if ret.get("overall") else []) or [0])
        best = max([s["val"] for s in c["by_season"].values()] or [0])
        rows.append((fits(pid), peak, best, pid))
    rows.sort(key=lambda x: (not x[0], -x[1], -x[2]))
    fit, peak, best, pid = rows[0]
    pos = (h.players.get(pid) or {}).get("position")
    peak_txt = f"peak {peak:.0f} overall" if peak else f"best season {best} value"
    return _answer(f"{h.name(pid)} ({pos}): {peak_txt}, at {('your weakest group' if fit and want else 'a position you could use')}.",
                   analysis=[f"Your weakest group against the league right now: {grp}.", "Prime = his best season-end rating, or his best season where ratings weren't kept."],
                   table=_tbl(["Player", "Pos", "Peak OVR", "Best season value", "Retired"], [[h.name(p), (h.players.get(p) or {}).get("position"), f"{k:.0f}" if k else "—", b, (h.players.get(p) or {}).get("retired_season")] for f, k, b, p in rows[:10]]))


def what_could_have_been(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.retired():
        pr = h.players[pid]
        age = int((pr.get("retirement") or {}).get("age") or 0)
        if age and age <= 32:
            rows.append((40 - age + h.career(pid)["val"] / 50, pid, f"retired at {age}"))
    for pid, d in _picks(h, rounds=(1, 1)):
        r = h.view.row(pid)
        if (r.get("age") or 0) >= 24 and d.get("draft_pot") and d["draft_pot"] - (r.get("pot") or 0) >= 6:
            rows.append(((d["draft_pot"] - r["pot"]) / 2, pid, f"projected {d['draft_pot']:.0f} at the draft, potential now {r['pot']:.0f}"))
        elif (r.get("age") or 0) >= 24 and d["overall"] <= 5 and (h.vs_slot(pid) or 0) <= -10:
            rows.append((-(h.vs_slot(pid) or 0) / 2, pid, f"#{d['overall']} pick, now {r.get('ovr', 0):.0f}"))
    if not rows:
        return _answer("No “what could have been” story yet: no early retirement and no top pick who fell well short.",
                       analysis=[f"The save has {h.span}."])
    rows.sort(key=lambda x: -x[0])
    _, pid, why = rows[0]
    return _answer(f"{h.name(pid)}: {why}.", analysis=["Early retirements (32 or younger) and top picks who fell well short of what scouts saw."],
                   table=_tbl(["Player", "Story"], [[h.name(p), w] for _, p, w in rows[:10]]), focus_ids=[pid] if pid in h.view.players else None)


def forgotten(h: Hist, raw: str, kind: str) -> Answer:
    """kind: legend (retired or long career, no awards), elite (top single seasons, no award), misunderstood, era, greatest_no_cup."""
    rows = []
    if kind == "elite":
        for sy in sorted(h.seasons):
            vals = sorted(((s["val"], pid) for pid in h.players for y, s in h.by_season(pid).items() if y == sy), reverse=True)[:5]
            for i, (v, pid) in enumerate(vals):
                if not h.awards_of(pid) and not (h.lg.get("hof") or {}).get(pid):
                    rows.append((v, pid, f"{v} value in {_season(sy)}, top {i + 1} in the league"))
        if not rows:
            return _answer("Every top-five season on record belongs to a player with a major award.")
    elif kind == "legend":
        pool = h.retired() or list(h.players)
        for pid in pool:
            c = h.career(pid)
            if h.awards_of(pid) or (h.lg.get("hof") or {}).get(pid) or not c["val"]:
                continue
            rows.append((c["val"] + h.nhl_imported(pid)["pts"] / 4, pid, f"{c['val']} value in {c['gp']} games, no major award" + (", retired" if pid in h.retired() else "")))
    elif kind == "era":
        for pid in h.players:
            c = h.career(pid)
            aw = h.awards_of(pid)
            if c["val"] or aw:
                rows.append((c["val"] + 40 * len(aw) + 25 * len(h.cups_of(pid)), pid,
                             f"{c['val']} value, {len(aw)} major award{'s' if len(aw) != 1 else ''}, {len(h.cups_of(pid))} Cup{'s' if len(h.cups_of(pid)) != 1 else ''}"))
    elif kind == "no_cup":
        for pid in h.players:
            if h.cups_of(pid):
                continue
            c, imp = h.career(pid), h.nhl_imported(pid)
            score = c["val"] + imp["pts"] + 2 * imp["w"]
            if score:
                rows.append((score + (h.ovr_now(pid) or 0), pid, f"{c['pts'] + imp['pts']} NHL points" + (f", {c['val']} value in this save" if c["val"] else "")))
    elif kind == "misunderstood":
        for sy, s in h.seasons.items():
            vals = sorted(((st["val"], pid) for pid in h.players for y, st in h.by_season(pid).items() if y == sy), reverse=True)
            order = [pid for _, pid in vals]
            for a in s.get("awards") or []:
                pid = a.get("player_id")
                if pid and pid in order and any(m in str(a.get("award") or "").lower() for m in ("hart", "lindsay")):
                    rows.append((order.index(pid), pid, f"won the {a['award']} in {_season(sy)} while {_ord(order.index(pid) + 1)} in league value"))
        if not rows:
            return forgotten(h, raw, "elite")
    if not rows:
        return _no_history(h, "careers")
    rows.sort(key=lambda x: -x[0])
    _, pid, why = rows[0]
    lead = {"legend": "Forgotten legend", "elite": "Least remembered", "era": "The player of this era",
            "no_cup": "Greatest without a Cup", "misunderstood": "Most misunderstood legacy"}[kind]
    notes = {"legend": "Most career value without a major award or Hall of Fame call; retired players first.",
             "elite": "Top-five seasons by value whose player never won a major award.",
             "era": "Career value, with 40 per major award and 25 per Cup.",
             "no_cup": "Real NHL points before the save count where imported; Cups are this save's only.",
             "misunderstood": "Hart and Lindsay winners whose season ranked lowest in league value."}[kind]
    return _answer(f"{lead}: {h.name(pid)}, {why}.", analysis=[notes, VALUE],
                   table=_tbl(["Player", "Why"], [[h.name(p), w] for _, p, w in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def hof_bad_luck(h: Hist, raw: str) -> Answer:
    rows = []
    sevens = Counter()
    for sy, ser in h.series():
        if _games(ser) == 7:
            sevens[(sy, _loser(ser))] += 1
    for pid in h.players:
        if h.cups_of(pid):
            continue
        peak = _peak(h, pid)
        c = h.career(pid)
        if peak < 90 and c["val"] < 200:
            continue
        heartbreaks = sum(n for (sy, t), n in sevens.items() if any(int(l.get("season") or 0) == sy and str(l.get("team_id")) == t for l in h.lines(pid)))
        rows.append((peak + c["val"] / 20 + 5 * heartbreaks, pid, peak, c["val"], heartbreaks))
    if not rows:
        return _answer("No Hall-of-Fame-calibre player has gone without a Cup yet.")
    rows.sort(key=lambda x: -x[0])
    _, pid, peak, v, hb = rows[0]
    return _answer(f"{h.name(pid)}: peaked at {peak:.0f} overall with {v} value, no Cup" + (f", and {hb} game-7 elimination{'s' if hb != 1 else ''}." if hb else "."),
                   analysis=["Elite peak (90+) or 200+ value, never a champion; game-7 exits add to the bad luck."],
                   table=_tbl(["Player", "Peak OVR", "Value", "Game-7 exits"], [[h.name(p), f"{k:.0f}", vv, b] for _, p, k, vv, b in rows[:10]]),
                   focus_ids=[pid] if pid in h.view.players else None)


def unexpected_hof(h: Hist, raw: str) -> Answer:
    hof = list((h.lg.get("hof") or {}).values())
    if not hof:
        rows = sorted(((h.career(p)["val"], p) for p in h.retired()), reverse=True)
        tail = f" Closest retired candidate: {h.name(rows[0][1])} ({rows[0][0]} value)." if rows and rows[0][0] else ""
        return _answer("Nobody has been inducted into the Hall of Fame in this save yet." + tail)
    def slot(x: Dict[str, Any]) -> int:
        d = h.drafts().get(str(x.get("player_id")))
        return d["overall"] if d else 300
    hof.sort(key=lambda x: -slot(x))
    x = hof[0]
    return _answer(f"{x.get('name')}: inducted {x.get('inducted_season')} ({'; '.join(x.get('reasons') or [])}) after {h.drafted(str(x.get('player_id')))}.",
                   analysis=["“Unexpected” = latest draft slot among inductees; undrafted counts as latest."],
                   table=_tbl(["Inductee", "Inducted", "Drafted", "Why"], [[y.get("name"), y.get("inducted_season"), h.pick_label(str(y.get("player_id"))), "; ".join(y.get("reasons") or [])] for y in hof]))


def face_unexpected(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        with_us = sum(h.val(pid, l) for l in h.lines(pid) if str(l.get("team_id")) == h.user)
        if with_us > 0:
            rows.append((with_us, pid))
    rows.sort(reverse=True)
    top = rows[:10]
    if not top:
        return _no_history(h, "franchise production")
    def slot(pid: str) -> int:
        d = h.drafts().get(pid)
        return d["overall"] if d else 300
    pick = max(top[:5], key=lambda x: (slot(x[1]), x[0]))
    v, pid = pick
    rank = [p for _, p in top].index(pid) + 1
    return _answer(f"{h.name(pid)}: {_ord(rank)} in franchise value in this save ({v}), and he went {h.drafted(pid)}.",
                   analysis=["Among your top five producers in the save, the one nobody would have picked: latest draft slot, undrafted counts as latest."],
                   table=_tbl(["Player", "Value with you", "Drafted"], [[h.name(p), vv, h.pick_label(p)] for vv, p in top]),
                   focus_ids=[pid] if pid in h.view.players else None)


# =========================================================================== seasons & records
def _team_seasons(h: Hist) -> List[Tuple[int, str, Dict[str, Any]]]:
    return [(sy, t, row) for sy, s in sorted(h.seasons.items()) for t, row in (s.get("standings") or {}).items() if row.get("gp")]


def collapse(h: Hist, raw: str) -> Answer:
    rows = []
    for sy, t, row in _team_seasons(h):
        before = h.pts(sy - 1, t)
        if before is not None:
            rows.append((before - int(row.get("pts") or 0), sy, t, before, int(row.get("pts") or 0)))
    if not rows:
        return _answer("A collapse needs two finished seasons to compare; this save has " + h.span + ".")
    rows.sort(key=lambda x: -x[0])
    d, sy, t, a, b = rows[0]
    return _answer(f"The {_season(sy)} {h.team(t)}: {a} points the season before, {b} that year (−{d}).",
                   table=_tbl(["Season", "Team", "Before", "After", "Drop"], [[_season(s), h.abbr(x), p, q, f"−{dd}"] for dd, s, x, p, q in rows[:10]]))


def perfection(h: Hist, raw: str) -> Answer:
    rows = [(int(r.get("pts") or 0) / (2 * int(r["gp"])), sy, t, r) for sy, t, r in _team_seasons(h)]
    if h.live:
        for t in h.view.teams:
            st = next((x for x in h.view.standings() if str(x.get("team_id")) == t), None)
            if st and st.get("gp"):
                rows.append((int(st.get("pts") or 0) / (2 * int(st["gp"])), h.view.season, t, {**st, "live": True}))
    if not rows:
        return _no_history(h, "seasons")
    rows.sort(key=lambda x: -x[0])
    pct, sy, t, r = rows[0]
    return _answer(f"The {_season(sy)} {h.team(t)}: {r.get('w')}-{r.get('l')}-{r.get('otl')}, {r.get('pts')} points, a {pct:.3f} points percentage{' (season in progress)' if r.get('live') else ''}.",
                   table=_tbl(["Season", "Team", "W-L-OTL", "Points", "Pts%"], [[_season(s), h.abbr(x), f"{q.get('w')}-{q.get('l')}-{q.get('otl')}", q.get("pts"), f"{p:.3f}"] for p, s, x, q in rows[:10]]))


def chaos(h: Hist, raw: str) -> Answer:
    rows = []
    for sy, s in sorted(h.seasons.items()):
        pts = [int(r.get("pts") or 0) for r in (s.get("standings") or {}).values() if r.get("gp")]
        if len(pts) < 8:
            continue
        spread = statistics.pstdev(pts)
        ups = sum(1 for y, ser in h.series() if y == sy and (h.pts(sy, _loser(ser)) or 0) > (h.pts(sy, str(ser["winner_id"])) or 0))
        trades = sum(1 for t in h.trades() if t["season"] == sy)
        rows.append((-spread + 3 * ups + trades / 10, sy, spread, ups, trades))
    if not rows:
        return _no_history(h, "seasons")
    rows.sort(key=lambda x: -x[0])
    _, sy, sp, ups, tr = rows[0]
    return _answer(f"{_season(sy)}: the tightest standings on record (points spread {sp:.1f}), {ups} playoff upset{'s' if ups != 1 else ''} and {tr} trades.",
                   analysis=["Chaos = tight standings, playoff upsets and trade volume."],
                   table=_tbl(["Season", "Points spread", "Upsets", "Trades"], [[_season(s), f"{p:.1f}", u, t] for _, s, p, u, t in rows]))


def season_talent(h: Hist, raw: str) -> Answer:
    rows = []
    for sy in sorted(h.seasons):
        vals = sorted((st["pts"] for pid in h.players if not h.goalie(pid) for y, st in h.by_season(pid).items() if y == sy), reverse=True)
        if len(vals) < 50:
            continue
        rows.append((statistics.mean(vals[:50]), sy, sum(1 for v in vals if v >= 100), sum(1 for v in vals if v >= 80)))
    if not rows:
        return _no_history(h, "seasons")
    rows.sort(key=lambda x: -x[0])
    avg, sy, c100, c80 = rows[0]
    return _answer(f"{_season(sy)}: the top 50 scorers averaged {avg:.0f} points, with {c100} at 100+ and {c80} at 80+.",
                   table=_tbl(["Season", "Top-50 avg pts", "100+ pts", "80+ pts"], [[_season(s), f"{a:.0f}", x, y] for a, s, x, y in rows]))


def team_seasons_records(h: Hist) -> List[Tuple[str, str, float, float, str]]:
    """(record, holder, value, runner-up value, season) for single-season player records."""
    stats = {"pts": "points", "g": "goals", "w": "goalie wins", "pim": "penalty minutes", "pm": "plus-minus"}
    out = []
    for key, label in stats.items():
        vals = []
        for pid in h.players:
            for y, s in h.by_season(pid).items():
                if key == "w" and not h.goalie(pid) or key != "w" and h.goalie(pid):
                    continue
                vals.append((s[key], pid, y))
        vals.sort(reverse=True)
        if len(vals) > 1 and vals[0][0]:
            out.append((f"{'best' if key == 'pm' else 'most'} {label} in a season", h.name(vals[0][1]), vals[0][0], vals[1][0], _season(vals[0][2])))
        if key == "pm" and vals:
            low = min(vals)
            out.append(("worst plus-minus in a season", h.name(low[1]), low[0], sorted(vals)[1][0], _season(low[2])))
    return out


def unbreakable(h: Hist, raw: str) -> Answer:
    recs = [r for r in team_seasons_records(h) if r[3] and r[2] > 0]
    if not recs:
        return _no_history(h, "records")
    recs.sort(key=lambda r: -(r[2] / r[3]))
    name, who, v, nxt, sy = recs[0]
    return _answer(f"{who}'s {v} for {name} ({sy}): the next best is {nxt}, {(v / nxt - 1) * 100:.0f}% behind.",
                   analysis=["Unbreakable = the biggest margin over the next-best mark."],
                   table=_tbl(["Record", "Holder", "Mark", "Next best", "Season"], [list(r) for r in recs]))


def weird_record(h: Hist, raw: str) -> Answer:
    rows = []
    for pid in h.players:
        if h.goalie(pid):
            continue
        for y, s in h.by_season(pid).items():
            if s["gp"] >= 70 and s["g"] == 0:
                rows.append((s["gp"], f"{h.name(pid)} played {s['gp']} games in {_season(y)} without scoring a goal"))
            if s["pim"] >= 150:
                rows.append((s["pim"] / 2, f"{h.name(pid)} took {s['pim']} penalty minutes in {_season(y)}"))
            if s["pm"] <= -40:
                rows.append((-s["pm"] * 2, f"{h.name(pid)} finished {_season(y)} at {s['pm']}"))
            if s["pts"] >= 60 and s["g"] <= 5:
                rows.append((s["pts"], f"{h.name(pid)} had {s['pts']} points in {_season(y)} with only {s['g']} goals"))
    if not rows:
        recs = team_seasons_records(h)
        if not recs:
            return _no_history(h, "records")
        r = min(recs, key=lambda x: x[2])
        return _answer(f"Nothing truly strange yet; the oddest standing mark is {r[1]}'s {r[2]} for {r[0]} ({r[4]}).")
    rows.sort(key=lambda x: -x[0])
    return _answer(f"{rows[0][1]}.", analysis=["Looks for 70-game goalless seasons, 150+ PIM, −40 or worse, and 60-point seasons with five goals or fewer."],
                   table=_tbl(["Oddity"], [[w] for _, w in rows[:10]]))


def _games_archive(h: Hist) -> List[Dict[str, Any]]:
    try:
        from services.sim_history import _read_games

        return [g for g in _read_games(h.session) if g.get("hg") is not None and g.get("ag") is not None]
    except Exception:
        return []


def _game_label(g: Dict[str, Any]) -> str:
    if g.get("home") and g.get("away"):
        tail = " (OT)" if g.get("ot") else " (SO)" if g.get("so") else ""
        return f"{g.get('date')}: {g['away']} {g['ag']} @ {g['home']} {g['hg']}{tail}"
    return str(g.get("summary") or g.get("date"))


def high_scoring_game(h: Hist, raw: str) -> Answer:
    games = _games_archive(h)
    if not games:
        return _answer("No game scores are archived yet.")
    games.sort(key=lambda g: -(int(g["hg"]) + int(g["ag"])))
    g = games[0]
    return _answer(f"{_game_label(g)}: {int(g['hg']) + int(g['ag'])} goals.",
                   analysis=[f"Searched {len(games)} archived games."],
                   table=_tbl(["Game", "Goals"], [[_game_label(x), int(x["hg"]) + int(x["ag"])] for x in games[:10]]))


def longest_game(h: Hist, raw: str) -> Answer:
    games = _games_archive(h)
    ot = [g for g in games if g.get("ot") or g.get("so")]
    po = [(sy, ser, g) for sy, ser, g in _po_games(h) if int(g.get("ot_periods") or 0) > 0]
    po += [(h.view.season, {}, e) for e in (getattr(h.session, "playoff_game_entries_by_gid", None) or {}).values()
           if isinstance(e, dict) and int(e.get("ot_periods") or 0) > 0]
    if po:
        sy, ser, e = max(po, key=lambda x: int(x[2].get("ot_periods") or 0))
        n = int(e["ot_periods"])
        return _answer(f"{_season(sy)} playoffs, Game {e.get('game')}: {h.team(e.get('away_id'))} {e.get('away_score')} @ {h.team(e.get('home_id'))} {e.get('home_score')}, decided in {_ord(n)} overtime.",
                       analysis=["Length is counted in overtime periods; regular-season overtime is one period at most."],
                       table=_tbl(["Season", "Game", "OT periods"], [[_season(a), f"{h.abbr(x.get('away_id'))} @ {h.abbr(x.get('home_id'))} G{x.get('game')}", x.get("ot_periods")]
                                                                      for a, _, x in sorted(po, key=lambda y: -int(y[2].get("ot_periods") or 0))[:10]]))
    if not ot:
        return _answer("No game has gone past regulation in the archive yet.")
    so = sum(1 for g in ot if g.get("so"))
    g = max(ot, key=lambda x: int(x["hg"]) + int(x["ag"]))
    return _answer(f"Game length isn't stored beyond overtime or shootout, so there's no single longest game. {len(ot)} of {len(games)} archived games went past regulation ({so} to a shootout).",
                   analysis=[f"Highest-scoring of them: {_game_label(g)}."],
                   table=_tbl(["Game"], [[_game_label(x)] for x in ot[-10:]]))


def expansion(h: Hist, raw: str) -> Answer:
    newest = [t for t in h.view.team_meta if h.abbr(t) in ("SEA", "VGK", "UTA")]
    rows = []
    for t in newest:
        pts = [(sy, h.pts(sy, t), h.rank(sy, t)) for sy in sorted(h.seasons)]
        made = [sy for sy, _, r in pts if r and r <= 16]
        rows.append((made[0] if made else 9999, t, pts))
    if not rows:
        return _answer("There are no expansion clubs in this save.")
    rows.sort()
    first, t, pts = rows[0]
    head = "No club has been added in this save; of the league's newest real-life franchises, "
    if first == 9999:
        return _answer(head + "none has finished in the top half yet.",
                       table=_tbl(["Team"] + [_season(sy) for sy in sorted(h.seasons)], [[h.abbr(x)] + [f"{p} pts (#{r})" for _, p, r in pp] for _, x, pp in rows]))
    return _answer(head + f"the {h.team(t)} were first into the top half ({_season(first)}).",
                   table=_tbl(["Team"] + [_season(sy) for sy in sorted(h.seasons)], [[h.abbr(x)] + [f"{p} pts (#{r})" for _, p, r in pp] for _, x, pp in rows]))


def overachieved(h: Hist, raw: str) -> Answer:
    return _need_playoffs(h) or improbable_run(h, raw, any_team=True)


def forgettable_champion(h: Hist, raw: str) -> Answer:
    rows = []
    for sy, t in h.champions():
        winners = sum(1 for a in h.seasons[sy].get("awards") or [] if a.get("player_id") and any(
            str(l.get("team_id")) == t and int(l.get("season") or 0) == sy for l in h.lines(str(a["player_id"]))))
        rows.append(((h.rank(sy, t) or 0) - 5 * winners, sy, t, h.rank(sy, t), winners))
    if not rows:
        return _no_history(h, "champions")
    rows.sort(key=lambda x: -x[0])
    _, sy, t, rank, aw = rows[0]
    return _answer(f"The {_season(sy)} {h.team(t)}: {_ord(rank)} in the regular season and {aw} individual award winner{'s' if aw != 1 else ''} on the roster.",
                   analysis=["Forgettable = weak regular season and no award winners to remember them by."],
                   table=_tbl(["Season", "Champion", "League rank", "Award winners"], [[_season(s), h.abbr(x), r, a] for _, s, x, r, a in rows]))


def wasted_talent(h: Hist, raw: str) -> Answer:
    rows = []
    for t in h.view.team_meta:
        gf = sum(int((h.seasons[sy].get("standings") or {}).get(t, {}).get("gf") or 0) for sy in h.seasons)
        pts = sum(h.pts(sy, t) or 0 for sy in h.seasons)
        wins = sum(1 for sy, ser in h.series() if str(ser["winner_id"]) == t)
        cups = sum(1 for _, c in h.champions() if c == t)
        rows.append((t, gf, pts, wins, cups))
    if not h.seasons:
        return _no_history(h, "seasons")
    by_pts = sorted(rows, key=lambda x: -x[2])
    # A high points rank (small index) with little playoff success sorts first.
    scored = sorted(((i + 3 * x[3] + 12 * x[4], x) for i, x in enumerate(by_pts)), key=lambda y: y[0])
    _, (t, gf, pts, wins, cups) = scored[0]
    return _answer(f"The {h.team(t)}: {pts} regular-season points across the save ({_ord(by_pts.index(next(r for r in by_pts if r[0] == t)) + 1)} in the league), but {wins} playoff series win{'s' if wins != 1 else ''} and {cups} Cup{'s' if cups != 1 else ''}.",
                   analysis=["Regular-season success that didn't turn into playoff series wins or Cups."],
                   table=_tbl(["Team", "Points", "Goals for", "Series won", "Cups"], [[h.abbr(x[0]), x[2], x[1], x[3], x[4]] for _, x in scored[:10]]))


def fans_change(h: Hist, raw: str) -> Answer:
    if h.series():
        out = upset(h, raw) if any((h.pts(sy, _loser(x)) or 0) > (h.pts(sy, str(x["winner_id"])) or 0) for sy, x in h.series()) else heartbreak(h, raw)
        out["answer"] = "The result fans would most like back: " + out["answer"][0].lower() + out["answer"][1:]
        return out
    out = robbery(h, raw)
    out["answer"] = "With no playoff heartbreak on record, the event fans would most want back is the most lopsided trade: " + out["answer"][0].lower() + out["answer"][1:]
    out.setdefault("analysis", []).append(_conf(MEDIUM, "the trade is exact; that fans would pick it is a judgement"))
    return out


def injury_ripple(h: Hist, raw: str) -> Answer:
    log = [e for e in list(getattr(h.session, "injury_log_major", None) or []) if isinstance(e, dict)]
    if not log:
        return _answer("Injuries aren't archived across seasons, so their effect on later Cups can't be traced.",
                       analysis=["This season's injury report is available: ask “injury report”."])
    def days(e: Dict[str, Any]) -> int:
        return int(e.get("days") or e.get("duration_days") or e.get("games") or 0)
    e = max(log, key=lambda x: days(x) * ((h.ovr_now(str(x.get("player_id") or "")) or 70) - 60))
    pid = str(e.get("player_id") or "")
    return _answer(f"{e.get('player_name') or h.name(pid)}: out {days(e)} days ({e.get('injury') or e.get('type') or 'injury'}), the costliest major injury on record by length and the player's rating.",
                   analysis=["Only this season's major injuries are kept, so effects on later Cups can't be traced yet."])


# =========================================================================== router
def _named(name: str, fn: Callable[[Hist, str], Answer]) -> Callable[[Hist, str], Answer]:
    fn.__name__ = name
    return fn


RULES: List[Tuple[str, Callable[[Hist, str], Answer]]] = [
    # trades
    (r"trade .*(produced|generated) the most|best trade (in|of) (franchise|our|team)|most (total )?(war|value) .*trade", best_trade_for_us),
    (r"trade .*looked (terrible|bad|awful).*(help|work)", terrible_then_good),
    (r"(ended|closed|killed) (our|the|their|[a-z' ]+?) (championship|cup|contention) window|window .*(end|close)", window_end),
    (r"value relative to (acquisition|acquiring) cost|value .*(acquisition|acquire) cost|bang for (the|our) buck", value_vs_cost),
    (r"greatest robbery|most lopsided trade|biggest (trade )?(robbery|fleecing)", robbery),
    (r"balance of power", balance_of_power),
    (r"benefit(ed|ted)? (the )?most from trading with|won .*trades? (with|against) (us|the )", benefited_from_us),
    (r"deadline (acquisition|pickup|addition|move)", deadline_impact),
    (r"traded the most (times)?|most traded player", traded_most),
    (r"(most )?(frequently|often) (exchanged|traded|moved) .*(pick|asset|player)|asset .*(most|frequently) (exchanged|traded)", most_exchanged),
    (r"superstar .*changed teams|star .*(moved|traded) .*unexpected", superstar_moved),
    (r"(team|club|franchise) .*(won|wins) the most trades|historically won .*trades", team_trade_record),
    (r"trade .*(led|lead) to a (stanley )?cup|cup winner .*without .*trade|won without a (specific )?trade|trade .*won (us|them) the cup", trade_to_cup),
    (r"what would .*(look like|have happened) if|if we (never|hadn't|had not) (made|done)", counterfactual),
    (r"trade offer .*(reject|turn(ed)? down|declin)|(rejected|declined) (trade )?offer", rejected_offer),
    (r"butterfly effect", butterfly),
    (r"single (event|moment) .*(altered|changed) .*history", single_event),
    (r"biggest mistake (this|our|the) (organi[sz]ation|franchise|team)|worst (move|decision) (in|of) franchise|biggest (regret|blunder)", _named("mistake", lambda h, raw: mistake(h, raw))),
    (r"gm decision", _named("mistake", lambda h, raw: mistake(h, raw))),
    (r"give up on too early|gave up on too early|let go too early|traded away too (early|soon)", gave_up_early),
    (r"improved the most after leaving|better after leaving (us|our)", _named("gave_up_early", lambda h, raw: gave_up_early(h, raw, improved=True))),
    # drafts
    (r"draft pick hurt us|pick .*hurt .*(selected|picked|taken) after|who (was|got) (selected|picked|taken) after", pick_regret),
    (r"(biggest|best) draft steal", _named("draft_steal", lambda h, raw: draft_steal(h, raw, team_only=True))),
    (r"draft class .*most (nhl )?games", _named("class_totals", lambda h, raw: class_totals(h, raw, "gp"))),
    (r"draft class .*most (total )?points", _named("class_totals", lambda h, raw: class_totals(h, raw, "pts"))),
    (r"first[- ]round pick .*(biggest )?bust|biggest (first[- ]round )?bust", first_round_bust),
    (r"late[- ]round pick", _named("draft_steal", lambda h, raw: draft_steal(h, raw, team_only=False, late=True))),
    (r"exceeded (scouting|scouts'?) expectations|beat (his )?(scouting|projection)", scouting_beat),
    (r"prospect .*failed|failed despite|untouchable .*(fail|bust)", _named("scouting_beat", lambda h, raw: scouting_beat(h, raw, failed=True))),
    (r"draft .*(future )?award winners", class_awards),
    (r"gap between draft position and (career )?success|outperformed (his )?draft (position|slot)", gap_draft_success),
    (r"strongest draft class|best draft class in (league )?history", strongest_class),
    (r"prospect becoming a bust|bust changed the standings", bust_standings),
    (r"draft lottery", lottery),
    (r"benefit(ed|ted)? the most from being drafted|drafted by (their|his) (specific )?organi[sz]ation", org_benefit),
    # playoffs
    (r"(biggest|greatest|largest) (playoff )?upset", upset),
    (r"(cup|stanley cup) champion .*weakest|weakest .*(cup )?champion", weakest_champion),
    (r"blew the biggest (series )?lead|blown (series )?lead|blew a \d-\d lead", blown_lead),
    (r"game ?(7|seven) goals|goals in game ?(7|seven)", game7_goals),
    (r"biggest goal|most important goal", biggest_goal),
    (r"playoff run .*improbable|most improbable (playoff )?run|cinderella", improbable_run),
    (r"goalie .*carried .*(playoff|team)|goalie .*furthest", goalie_carry),
    (r"closest (cup |stanley cup )?final", closest_final),
    (r"overachieved .*(projection|expect)|playoff team overachieved", overachieved),
    (r"heartbreaking elimination|most painful (loss|elimination)", heartbreak),
    (r"dynasty .*(closest|almost|nearly) .*never|closest to never happening", dynasty_close),
    (r"season .*(started|began|kicked off) (our|the|their|[a-z' ]+?) (dynasty|best run|run)|start(ed)? of (our|the|their) dynasty", dynasty_start),
    (r"most (alternate[- ]history )?[\"“]?what[- ]if", most_what_ifs),
    # careers
    (r"strangest career|weirdest career|oddest career", strangest_path),
    (r"declined the fastest|fell off the fastest|fastest decline", declined_fastest),
    (r"one[- ]season wonder|one[- ]hit wonder", one_season_wonder),
    (r"best career .*(without|never) (winning )?(a )?(major )?award", best_no_award),
    (r"longest (nhl )?career", longest_career),
    (r"biggest overall[- ]rating increase|biggest (rating|overall) (jump|increase|gain)|improved the most in history", biggest_rise),
    (r"unexpected breakout|biggest breakout", breakout),
    (r"consistently underrated|underrated throughout", underrated_career),
    (r"retired player .*(improve|help|fit)|bring(ing)? back .*prime", retired_prime),
    (r"what could have been", what_could_have_been),
    (r"face of the [a-z' ]*franchise .*(nobody|no one|unexpected)|unexpected(ly)? .*face of the franchise", face_unexpected),
    (r"unexpectedly became a hall of famer|unexpected hall of famer|surprise hall of famer", unexpected_hof),
    (r"hall[- ]of[- ]fame talent .*(luck|unlucky)|terrible luck", hof_bad_luck),
    (r"forgotten legend", _named("forgotten", lambda h, raw: forgotten(h, raw, "legend"))),
    (r"least remembered|forgotten despite", _named("forgotten", lambda h, raw: forgotten(h, raw, "elite"))),
    (r"legacy is (most )?misunderstood|misunderstood legacy", _named("forgotten", lambda h, raw: forgotten(h, raw, "misunderstood"))),
    (r"defined an (entire )?era|player of (the|this) era", _named("forgotten", lambda h, raw: forgotten(h, raw, "era"))),
    (r"greatest player never to win a cup|best player (never|without) (to )?win(ning)? (a|the) cup", _named("forgotten", lambda h, raw: forgotten(h, raw, "no_cup"))),
    # seasons & records
    (r"weirdest record|strangest record", weird_record),
    (r"unbreakable record", unbreakable),
    (r"highest[- ]scoring game", high_scoring_game),
    (r"longest game", longest_game),
    (r"season .*(most chaos|most chaotic)|most chaotic season", chaos),
    (r"(greatest|biggest|worst) (single[- ]season )?collapse", collapse),
    (r"expansion (franchise|team|club)", expansion),
    (r"season produced the most .*talent|most talented season", season_talent),
    (r"closest to perfection|most dominant (regular )?season|best regular season ever", perfection),
    (r"championship team .*forgotten|forgotten champion", forgettable_champion),
    (r"wasted the most talent", wasted_talent),
    (r"fans could change one event|change one event in (league )?history", fans_change),
    (r"injury .*ripple|injury .*(changed|altered) .*(championship|cup|history)", injury_ripple),
]


# =========================================================================== reasoning layer
# When the direct record is empty, answer the nearest real question instead of stopping,
# say which evidence it rests on, and how sure the answer is.
HIGH, MEDIUM, LOW = "high", "medium", "low"


def _conf(level: str, why: str) -> str:
    return f"Confidence: {level}, {why}."


def _how_acquired(h: Hist, pid: str) -> str:
    d = h.drafts().get(pid)
    if d and d.get("team_id") == h.user:
        return f"drafted {h.pick_label(pid)}"
    for t in h.trades():
        if t["user"] and any(p == pid for p, _, _ in t["recv"].get(h.user, [])):
            return f"acquired by trade {h.on(t)}"
    return "signed or on the roster when the save began"


def _franchise_core(h: Hist, tid: Optional[str] = None, season: Optional[int] = None) -> List[Tuple[int, str]]:
    tid = tid or h.user
    rows = []
    for pid in h.players:
        v = sum(h.val(pid, l) for l in h.lines(pid) if str(l.get("team_id")) == tid and (season is None or int(l.get("season") or 0) == season))
        if v > 0:
            rows.append((v, pid))
    return sorted(rows, reverse=True)


def mistake(h: Hist, raw: str) -> Answer:
    """Biggest mistake: every kind of decision is a candidate, scored on one 0-100 severity scale."""
    cands: List[Tuple[float, str, str, str]] = []  # severity, category, story, confidence
    for t in h.trades():
        if t["user"] and h.net(t, h.user) < 0:
            cost = -h.net(t, h.user)
            cands.append((min(100, cost / 2), "trade", f"the {h.when(t)} trade ({h.moved(t)}), which has cost you {cost} value since", HIGH))
    by_year: Dict[int, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for pid, d in h.drafts().items():
        if pid in h.view.players and d.get("year"):
            by_year[d["year"]].append((pid, d))
    for year, picks in by_year.items():
        for pid, d in picks:
            if d.get("team_id") != h.user:
                continue
            mine = h.ovr_now(pid) or 0
            later = [(h.ovr_now(p) or 0, p) for p, dd in picks if d["overall"] < dd["overall"] <= d["overall"] + 32]
            if later:
                best, other = max(later)
                gap = best - mine
                if gap > 0:
                    young = max(h.view.row(pid).get("age") or 0, h.view.row(other).get("age") or 0) < 22
                    story = (f"drafting {h.name(pid)} at {h.pick_label(pid)}. At the time the pick was defensible, but {h.name(other)} was still on the board, "
                             f"went {h.drafts()[other]['overall'] - d['overall']} picks later to the {h.team(h.drafts()[other].get('team_id'))}, "
                             f"and is already {gap:.0f} overall better")
                    cands.append((gap * 3 * (0.6 if young else 1), "draft", story, LOW if young else MEDIUM))
            gap = h.vs_slot(pid)
            if d.get("round") == 1 and gap is not None and gap < -4 and (h.view.row(pid).get("age") or 0) >= 21:
                cands.append((-gap * 4, "draft", f"spending a first-round pick ({h.pick_label(pid)}) on {h.name(pid)}, now {gap:+.0f} against that slot", MEDIUM))
    try:
        from .derived import market
    except Exception:
        market = None
    for r in h.view.team_rows(h.user):
        c = r.get("contract") or {}
        mv = market(h.view, r["id"]) if market else None
        if mv is not None and c.get("cap_hit_m"):
            over = float(c["cap_hit_m"]) - float(mv)
            if over > 1.0:
                yrs = max(1, int(c.get("years_left") or 1))
                cands.append((min(100, over * yrs * 6), "contract",
                              f"{r['name']}'s contract: ${c['cap_hit_m']:.2f}M against a market value near ${mv:.2f}M, with {yrs} year{'s' if yrs != 1 else ''} left", MEDIUM))
    for score, pid, gu, vu, ga, va, age, tid in _departures(h):
        if va > 0:
            cands.append((min(100, va / 2), "departure", f"letting {h.name(pid)} go: {va} value in {ga} games for the {h.team(tid)} since", MEDIUM))
    for pid, d in h.drafts().items():
        r = h.view.row(pid) if pid in h.view.players else {}
        if d.get("team_id") == h.user and r and r.get("team_id") == h.user and (r.get("age") or 0) >= 23 and (r.get("pot") or 0) - (r.get("ovr") or 0) >= 10:
            cands.append(((r["pot"] - r["ovr"]) * 0.8, "development", f"{r['name']}'s development: {r['ovr']:.0f} overall at {r['age']} against {r['pot']:.0f} potential", LOW))
    looked = "trades, draft picks and the players taken after them, contracts against market value, players who left, and stalled prospects"
    if not cands:
        return _answer("There's no franchise-defining mistake in this save yet: no trade has lost value, no pick has been passed by a better player, "
                       "no contract sits well above market and nobody you let go has made you regret it.",
                       analysis=[f"Looked at {looked}.", _conf(HIGH, "every category was checked and came back clean")])
    cands.sort(key=lambda x: -x[0])
    sev, cat, story, conf = cands[0]
    small = sev < 25
    lead = ("There isn't an obvious franchise-defining mistake yet, but the closest candidate is " if small else "The biggest mistake on record is ")
    tail = {"draft": " If that gap keeps growing, it becomes the organization's biggest missed opportunity.",
            "trade": " Nothing else you've done has cost as much.",
            "contract": " It limits what the cap can buy every year it runs.",
            "departure": " It's value that could have stayed in your lineup.",
            "development": " It's the clearest case of talent not turning into a player."}[cat]
    others = []
    seen = {cat}
    for c in cands[1:]:
        if c[1] not in seen:
            seen.add(c[1])
            others.append(f"{c[1].title()}: {c[2]}.")
        if len(others) == 2:
            break
    return _answer(lead + story + "." + tail,
                   analysis=others + [f"Weighed {looked}.", _conf(LOW if small else conf, "a small gap so far" if small else f"based on {cat} evidence")],
                   table=_tbl(["Kind", "Candidate", "Severity"], [[c[1], c[2], f"{c[0]:.0f}"] for c in cands[:8]]))


def _departures(h: Hist) -> List[Tuple[float, str, int, int, int, int, Optional[int], Any]]:
    out = []
    for pid in h.players:
        mine = [ln for ln in h.lines(pid) if str(ln.get("team_id")) == h.user]
        if not mine:
            continue
        last_us = max(int(ln.get("season") or 0) for ln in mine)
        after = [ln for ln in h.lines(pid) if int(ln.get("season") or 0) >= last_us and str(ln.get("team_id") or "") not in ("", h.user)]
        if not after:
            continue
        age_now = h.view.row(pid).get("age") or 0
        out.append((sum(h.val(pid, l) for l in after), pid, sum(int(l.get("gp") or 0) for l in mine), sum(h.val(pid, l) for l in mine),
                    sum(int(l.get("gp") or 0) for l in after), sum(h.val(pid, l) for l in after),
                    age_now - (h.view.season - last_us) if age_now else None, after[-1].get("team_id")))
    return sorted(out, key=lambda x: -x[0])


# ---- fallbacks: (question handler name) -> inference when the direct record is empty
def fb_built_within(h: Hist, raw: str) -> Optional[Answer]:
    core = _franchise_core(h)
    if not core:
        return None
    v, pid = core[0]
    return _answer(f"You haven't made a trade in this save, so the franchise's best acquisitions came another way. The most valuable is {h.name(pid)}, "
                   f"{_how_acquired(h, pid)}, with {v} value in your uniform.",
                   analysis=["With no trades, the nearest question is which acquisition paid off most.", _conf(MEDIUM, "no trades to judge, so this ranks every route a player came in by")],
                   table=_tbl(["Player", "Value with you", "How he arrived"], [[h.name(p), x, _how_acquired(h, p)] for x, p in core[:8]]))


def fb_watch_list(h: Hist, raw: str) -> Optional[Answer]:
    rows = [r for r in h.view.team_rows(h.user) if (r.get("contract") or {}).get("years_left", 9) <= 1 and r.get("age", 99) <= 27]
    rows.sort(key=lambda r: -(r.get("pts") or 0) - (r.get("ovr") or 0))
    if not rows:
        return _answer("No one has left your club since the save began, so there's no one you gave up on yet, and no young regular is about to reach free agency.",
                       analysis=[_conf(HIGH, "no departures and no young expiring contracts")])
    r = rows[0]
    return _answer(f"Nobody has left your club yet, so there's no regret on record. The player you'd most likely look back on is {r['name']} "
                   f"({r['age']}, {r['ovr']:.0f} overall), whose deal expires after this season.",
                   analysis=["Nearest question: which young regular could walk next.", _conf(LOW, "a forecast, not a record")],
                   rows=rows[:6], extra=("pts", "cap", "status"))


def fb_young_first_rounders(h: Hist, raw: str) -> Optional[Answer]:
    team = h.user if _ours(raw) else None
    picks = [(h.vs_slot(pid), pid) for pid, d in _picks(h, team, (1, 1)) if h.vs_slot(pid) is not None]
    if not picks:
        return None
    x, pid = min(picks)
    return _answer(f"No first-rounder is old enough to call a bust, but the early warning is {h.name(pid)} ({h.pick_label(pid)}): "
                   f"{h.ovr_now(pid):.0f} overall at {h.view.row(pid).get('age')}, {x:+.0f} against what that slot usually becomes.",
                   analysis=[DRAFT_RULE, _conf(LOW, "young players still move a lot")], table=_pick_rows(h, sorted(picks)), focus_ids=[pid])


def fb_deadline_regular(h: Hist, raw: str) -> Optional[Answer]:
    rows = []
    for t in h.trades():
        if t["month"] not in DEADLINE_MONTHS and t["iso"]:
            continue
        for to, items in t["recv"].items():
            for pid, name, v in items:
                if pid and v:
                    rows.append((v, name, to, t))
    if not rows:
        return None
    rows.sort(key=lambda x: -x[0])
    v, name, to, t = rows[0]
    dated = bool(t["iso"])
    return _answer(f"Playoff games for deadline pickups aren't on record, so the best measure is what they did after the move: {name}, "
                   f"picked up by the {h.team(to)} {h.on(t)}, has {v} value for them since.",
                   analysis=["Deadline = trades made in February or March." if dated else "Trade dates weren't kept on this save, so every in-season trade is a candidate.",
                             _conf(LOW, "regular-season value stands in for playoff impact")],
                   table=_tbl(["Player", "Team", "When", "Value since"], [[n, h.abbr(x), h.when(tr), vv] for vv, n, x, tr in rows[:10]]))


def fb_cup_core(h: Hist, raw: str) -> Optional[Answer]:
    if not h.champions():
        return None
    sy, champ = h.champions()[-1]
    core = _franchise_core(h, champ, sy)[:6]
    if not core:
        return None
    how = Counter("draft" if "drafted" in _how_acquired_for(h, p, champ) else "trade" if "trade" in _how_acquired_for(h, p, champ) else "other" for _, p in core)
    v, pid = core[0]
    built = (f"{how['draft']} of their top six were drafted and {how['trade']} traded for" if how["draft"] or how["trade"]
             else "none of their top six came by trade; the core was signed or already there when the save began")
    return _answer(f"No Cup on record turned on a trade. The {_season(sy)} {h.team(champ)} leaned on {h.name(pid)} ({v} value), and {built}.",
                   analysis=["Nearest question: how the most recent champion was built.", _conf(MEDIUM, "trades were checked first and none fed a Cup")],
                   table=_tbl(["Player", "Value in Cup season", "Arrived"], [[h.name(p), x, _how_acquired_for(h, p, champ)] for x, p in core]))


def _how_acquired_for(h: Hist, pid: str, tid: str) -> str:
    d = h.drafts().get(pid)
    if d and d.get("team_id") == tid:
        return f"drafted {h.pick_label(pid)}"
    if any(p == pid for t in h.trades() for p, _, _ in t["recv"].get(tid, [])):
        return "trade"
    return "signed or original roster"


def _best_non_champions(h: Hist) -> List[Tuple[int, int, str]]:
    rows = []
    champs = dict(h.champions())
    for sy, s in h.seasons.items():
        st = s.get("standings") or {}
        for t, r in st.items():
            if t != champs.get(sy):
                rows.append((int(r.get("pts") or 0), sy, t))
    return sorted(rows, reverse=True)


def fb_heartbreak(h: Hist, raw: str) -> Optional[Answer]:
    rows = _best_non_champions(h)
    if not rows:
        return None
    pts, sy, t = rows[0]
    rank = h.rank(sy, t)
    champ = dict(h.champions()).get(sy)
    return _answer(f"The brackets weren't kept, so the exact elimination isn't known, but the most painful season on record is the {_season(sy)} {h.team(t)}: "
                   f"{pts} points, {_ord(rank)} in the league, and no Cup" + (f" while the {h.team(champ)} won it." if champ else "."),
                   analysis=["Nearest evidence: the best regular seasons that ended without a title.", _conf(LOW, "regular-season record stands in for the playoff series")],
                   table=_tbl(["Season", "Team", "Points", "Rank"], [[_season(y), h.abbr(x), p, h.rank(y, x)] for p, y, x in rows[:10]]))


def fb_upset(h: Hist, raw: str) -> Optional[Answer]:
    champs = h.champions()
    if not champs:
        return None
    ranked = sorted(((h.rank(sy, t) or 1, sy, t) for sy, t in champs), reverse=True)
    rank, sy, t = ranked[0]
    if rank > 1:
        return _answer(f"The clearest upset on record is the {_season(sy)} {h.team(t)} winning the Cup from {_ord(rank)} in the league.",
                       analysis=["Brackets weren't kept, so earlier-round upsets are unknown.", _conf(MEDIUM, "champion and standings are exact")])
    pts, ysy, top = _best_non_champions(h)[0]
    return _answer(f"The favourite won the Cup every year on record, so there's no title upset. Brackets weren't kept, so earlier-round upsets are unknown; "
                   f"the best team that fell short was the {_season(ysy)} {h.team(top)} ({pts} points).",
                   analysis=[_conf(LOW, "only champions and standings survive for these seasons")])


def _best_season(h: Hist, tid: str) -> Optional[int]:
    ys = [sy for sy in h.seasons if h.pts(sy, tid) is not None]
    return max(ys, key=lambda y: h.pts(y, tid) or 0) if ys else None


def fb_champion_goalie(h: Hist, raw: str) -> Optional[Answer]:
    if _ours(raw) and h.user not in dict((t, 1) for _, t in h.champions()):
        sy = _best_season(h, h.user)
        goalies = [(sum(int(l.get("w") or 0) for l in h.lines(p) if int(l.get("season") or 0) == sy and str(l.get("team_id")) == h.user), p)
                   for p in h.players if h.goalie(p)] if sy else []
        goalies = sorted(g for g in goalies if g[0])
        if not goalies:
            return None
        w, pid = goalies[-1]
        return _answer(f"Their playoff goalie lines weren't kept, so the likeliest answer is the starter in the {h.team(h.user)}' best season: "
                       f"{h.name(pid)}, {w} wins in {_season(sy)} ({h.pts(sy, h.user)} points, {_ord(h.rank(sy, h.user))} in the league).",
                       analysis=[_conf(LOW, "the starter of their best regular season stands in for their deepest run")], focus_ids=[pid] if pid in h.view.players else None)
    rows = []
    for sy, champ in h.champions():
        for pid in h.players:
            if not h.goalie(pid):
                continue
            w = sum(int(l.get("w") or 0) for l in h.lines(pid) if int(l.get("season") or 0) == sy and str(l.get("team_id")) == champ)
            if w:
                rows.append((w, pid, sy, champ))
    if not rows:
        return None
    rows.sort(reverse=True)
    w, pid, sy, champ = rows[0]
    return _answer(f"Playoff goalie stats weren't kept for these seasons, so the likeliest answer is the champion's starter: {h.name(pid)}, "
                   f"{w} regular-season wins for the {_season(sy)} {h.team(champ)} before they won the Cup.",
                   analysis=[_conf(LOW, "the champion's starter in the regular season stands in for the playoff run")],
                   table=_tbl(["Goalie", "Season", "Champion", "Reg. season W"], [[h.name(p), _season(s), h.abbr(c), x] for x, p, s, c in rows[:8]]))


def fb_big_goal(h: Hist, raw: str) -> Optional[Answer]:
    team = h.user if _ours(raw) else None
    cups = [(sy, t) for sy, t in h.champions() if not team or t == team]
    if not cups and team:
        # No Cup: the biggest goals came in their best season; its top goal scorer is the likeliest owner.
        sy = _best_season(h, team)
        cups = [(sy, team)] if sy else []
    if not cups:
        return None
    rows = []
    for sy, t in cups:
        for pid in h.players:
            g = sum(int(l.get("g") or 0) for l in h.lines(pid) if int(l.get("season") or 0) == sy and str(l.get("team_id")) == t)
            if g:
                rows.append((g, pid, sy, t))
    if not rows:
        return None
    rows.sort(reverse=True)
    g, pid, sy, t = rows[0]
    won = dict(h.champions()).get(sy) == t
    return _answer(f"Individual playoff goals weren't kept for these seasons, so the exact goal is unknown. The likeliest owner is {h.name(pid)}, "
                   f"the top scorer ({g} goals) on the {_season(sy)} {h.team(t)}, " + ("who won the Cup." if won else "the club's best season on record."),
                   analysis=["Cup-winning, overtime and clinching goals are what count most; they're kept for playoffs from now on.", _conf(LOW, "the Cup team's top goal scorer is the most likely scorer of its biggest goals")],
                   table=_tbl(["Player", "Cup season", "Team", "Goals that season"], [[h.name(p), _season(s), h.abbr(x), gg] for gg, p, s, x in rows[:8]]))


def fb_clutch(h: Hist, raw: str) -> Optional[Answer]:
    c: Counter = Counter()
    for sy, t in h.champions():
        for pid in h.players:
            c[pid] += sum(int(l.get("g") or 0) for l in h.lines(pid) if int(l.get("season") or 0) == sy and str(l.get("team_id")) == t)
    c = Counter({k: v for k, v in c.items() if v})
    if not c:
        return None
    rows = c.most_common(8)
    pid, g = rows[0]
    return _answer(f"Game 7 scoring wasn't kept for these playoffs, so it can't be counted. The closest proxy is goals for Cup-winning teams: {h.name(pid)} leads with {g} "
                   f"in championship seasons, the player most often on the ice when it mattered.",
                   analysis=["Game 7 goals are recorded for every playoff from now on.", _conf(LOW, "championship-season scoring stands in for Game 7s")],
                   table=_tbl(["Player", "Goals in Cup seasons"], [[h.name(p), x] for p, x in rows]))


def fb_closest_final(h: Hist, raw: str) -> Optional[Answer]:
    rows = []
    for sy, champ in h.champions():
        st = h.seasons[sy].get("standings") or {}
        others = sorted((int(r.get("pts") or 0) for t, r in st.items() if t != champ), reverse=True)
        if others:
            rows.append(((h.pts(sy, champ) or 0) - others[0], sy, champ))
    if not rows:
        return None
    rows.sort(key=lambda x: abs(x[0]))
    gap, sy, champ = rows[0]
    return _answer(f"Final scores weren't kept for these seasons. The closest title race on record was {_season(sy)}: the {h.team(champ)} won the Cup with a "
                   f"{abs(gap)}-point {'edge over' if gap >= 0 else 'deficit to'} the next-best team.",
                   analysis=[_conf(LOW, "the regular-season margin stands in for the Final")])


def fb_overachieved(h: Hist, raw: str) -> Optional[Answer]:
    raw_rows = []
    for sy, t, r in _team_seasons(h):
        gf, ga, gp = int(r.get("gf") or 0), int(r.get("ga") or 0), int(r.get("gp") or 0)
        if gf and ga and gp:
            raw_rows.append((sy, t, int(r.get("pts") or 0), 2 * gp * gf ** 2 / (gf ** 2 + ga ** 2)))
    if not raw_rows:
        return None
    # Loser points lift every team above the pure win formula; centre it on the league.
    lift = statistics.mean(p - e for _, _, p, e in raw_rows)
    rows = [(p - (e + lift), sy, t, p, e + lift) for sy, t, p, e in raw_rows]
    rows.sort(key=lambda x: -x[0])
    d, sy, t, pts, exp = rows[0]
    return _answer(f"Playoff projections weren't kept, so the nearest measure is the regular season: the {_season(sy)} {h.team(t)} earned {pts} points "
                   f"where their goal differential usually earns about {exp:.0f}, {d:+.0f} above it.",
                   analysis=["Expected points from goals for and against (Pythagorean record).", _conf(MEDIUM, "standard method, regular season only")],
                   table=_tbl(["Season", "Team", "Points", "Expected", "Above"], [[_season(s), h.abbr(x), p, f"{e:.0f}", f"{dd:+.0f}"] for dd, s, x, p, e in rows[:10]]))


def fb_dynasty(h: Hist, raw: str) -> Optional[Answer]:
    cups = Counter(t for _, t in h.champions())
    dyn = [t for t, n in cups.items() if n >= 2]
    if not dyn:
        return None
    t = max(dyn, key=lambda x: cups[x])
    rows = []
    for sy, c in h.champions():
        if c != t:
            continue
        st = h.seasons[sy].get("standings") or {}
        nxt = max(int(r.get("pts") or 0) for x, r in st.items() if x != t)
        r = st.get(t) or {}
        rows.append(((h.pts(sy, t) or 0) - nxt, sy, int(r.get("gf") or 0) - int(r.get("ga") or 0)))
    rows.sort()
    gap, sy, gd = rows[0]
    return _answer(f"The {h.team(t)} ({cups[t]} Cups). Their series weren't archived, but the weakest link was {_season(sy)}: "
                   f"{'only ' if gap < 10 else ''}{gap} points clear of the next-best team (goal differential {gd:+d}). That's the season the run was most exposed.",
                   analysis=[_conf(LOW, "regular-season margins stand in for how close each playoff run came")],
                   table=_tbl(["Cup season", "Margin over next team", "Goal diff"], [[_season(s), f"{g:+d}", f"{d:+d}"] for g, s, d in rows]))


def fb_what_could(h: Hist, raw: str) -> Optional[Answer]:
    rows = [((r.get("pot") or 0) - (r.get("ovr") or 0), r) for r in h.view.rows() if (r.get("age") or 0) >= 24 and r.get("level") in ("NHL", "AHL")]
    rows = [x for x in rows if x[0] >= 8]
    if not rows:
        return None
    rows.sort(key=lambda x: -x[0])
    gap, r = rows[0]
    return _answer(f"No career has ended early yet, so the closest story is {r['name']} ({r['team']}): {r['ovr']:.0f} overall at {r['age']} with {r['pot']:.0f} potential, "
                   f"talent that hasn't arrived yet.",
                   analysis=[_conf(LOW, "an unfinished story; he could still close the gap")], rows=[x[1] for x in rows[:8]])


def fb_hof_track(h: Hist, raw: str) -> Optional[Answer]:
    rows = []
    for pid in h.view.players:
        c = h.career(pid)
        if c["seasons"] >= 2:
            pace = c["val"] / c["seasons"]
            d = h.drafts().get(pid)
            rows.append((pace + (d["overall"] / 10 if d else 25), pid, pace))
    if not rows:
        return None
    rows.sort(reverse=True)
    _, pid, pace = rows[0]
    return _answer(f"Nobody has been inducted yet. The likeliest surprise inductee is {h.name(pid)}: {h.drafted(pid)}, and producing {pace:.0f} value a season.",
                   analysis=["Ranked by production pace with extra weight for a late or no draft slot.", _conf(LOW, "a projection, not an induction")],
                   focus_ids=[pid])


FALLBACKS: Dict[str, Callable[[Hist, str], Optional[Answer]]] = {
    "best_trade_for_us": fb_built_within, "benefited_from_us": fb_built_within, "terrible_then_good": fb_built_within,
    "gave_up_early": fb_watch_list, "first_round_bust": fb_young_first_rounders, "scouting_beat": fb_young_first_rounders,
    "deadline_impact": fb_deadline_regular, "trade_to_cup": fb_cup_core,
    "upset": fb_upset, "fans_change": fb_heartbreak, "heartbreak": fb_heartbreak, "blown_lead": fb_heartbreak,
    "goalie_carry": fb_champion_goalie, "biggest_goal": fb_big_goal, "game7_goals": lambda h, raw: fb_clutch(h, raw),
    "closest_final": fb_closest_final, "overachieved": fb_overachieved, "dynasty_close": fb_dynasty,
    "what_could_have_been": fb_what_could, "unexpected_hof": fb_hof_track,
}
EMPTY = re.compile(r"^(No |Nobody|None of|You haven't|Game-by-game|Your club hasn't)")

# Slang and shorthand, rewritten before matching so the rules see the intended question.
SLANG: List[Tuple[str, str]] = [
    (r"\bwat\b|\bwhut\b", "what"), (r"\bwhos\b", "who is"), (r"\bare (biggest|worst|best)\b", r"our \1"),
    (r"\b(screw ?up|screwup|blunder|fumble|l\b|biggest l)\b", "mistake"),
    (r"\bbiggest mistake\b|\bworst mistake\b|\bour mistake\b|\bmistake\b", "biggest mistake this organization"),
    (r"shouldn'?t (of|have) (traded|let go|dealt)|regret (trading|losing|letting)", "give up on too early"),
    (r"\bclutchest\b|\bbiggest goal ever\b", "biggest goal"),
]


def _normalize(q: str) -> str:
    out = q
    for pat, rep in SLANG:
        out = re.sub(pat, rep, out)
    return out


def match(raw_question: str) -> Optional[Callable[[Hist, str], Answer]]:
    q = " ".join(str(raw_question or "").lower().replace("’", "'").split())
    for text in (q, _normalize(q)):
        for pattern, fn in RULES:
            if re.search(pattern, text):
                return fn
    return None


def _for_team(h: Hist, tid: str) -> Hist:
    """The same history read from another club's side: every “our/your” question is about it."""
    other = copy.copy(h)
    other.user, other._trades = tid, None
    return other


def _third_person(out: Answer, name: str) -> Answer:
    subs = [(r"\bYou haven't\b", f"The {name} haven't"), (r"\byou haven't\b", f"the {name} haven't"),
            (r"\bYou've\b", "They've"), (r"\byou've\b", "they've"), (r"\byou're\b", "they're"),
            (r"\b(for|with|against|from|to|cost|left|beat|joined) you\b", r"\1 them"), (r"\bYour\b", "Their"), (r"\byour\b", "their"),
            (r"\bYou\b", "They"), (r"\byou\b", "they")]

    def fix(text: Any) -> Any:
        if not isinstance(text, str):
            return text
        for pat, rep in subs:
            text = re.sub(pat, rep, text)
        return text

    out["answer"] = fix(out.get("answer"))
    out["analysis"] = [fix(a) for a in out.get("analysis") or []]
    if isinstance(out.get("table"), dict):
        out["table"]["cols"] = [fix(c) for c in out["table"].get("cols") or []]
    return out


def run(fn: Callable[[Hist, str], Answer], view: LiveView, raw_question: str, team: str = "") -> Answer:
    raw = " ".join(str(raw_question or "").lower().replace("’", "'").split())
    h = _hist(view)
    cpu = team and team != view.user_tid and team in view.team_meta
    if cpu:
        # A named club makes the question about that club's history, read from its side.
        h, raw = _for_team(h, team), raw + " franchise"
    out = fn(h, raw)
    fb = FALLBACKS.get(fn.__name__)
    if fb and EMPTY.match(str(out.get("answer") or "")):
        try:
            better = fb(h, raw)
        except Exception:
            better = None
        if better:
            better.setdefault("analysis", []).append("Direct record: " + str(out.get("answer")))
            out = better
    said = " ".join([str(out.get("answer") or "")] + list(out.get("analysis") or []))
    if "Confidence:" not in said:
        empty = EMPTY.match(str(out.get("answer") or ""))
        out.setdefault("analysis", []).append(_conf(HIGH, "nothing on record to infer from" if empty else "read straight from the save's records"))
    if h.seasons and len(h.seasons) < 3 and h.span not in said:
        out.setdefault("analysis", []).append(f"History is short: the save has {h.span}.")
    if cpu:
        out = _third_person(out, h.team(team))
        out.setdefault("analysis", []).insert(0, f"Read from the {h.team(team)}' side.")
    return out
