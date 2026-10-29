"""Analytics check: every answer about players or clubs is weighed against expected metrics.

After a layer answers, this adds one or two lines that test the answer against the
underlying numbers the sim records — individual expected goals (ixG), on-ice
expected-goal share (xGF%), shot-attempt share (CF%), goals saved above expected
(GSAx) and PDO — and says whether the evidence supports the headline or warns it may
be luck. It never changes the answer; it shows how much to trust it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .data import LiveView


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _st(view: LiveView, pid: str) -> Dict[str, Any]:
    st = view.stats.get(pid) if isinstance(view.stats, dict) else None
    return st if isinstance(st, dict) else {}


def player_line(view: LiveView, pid: str) -> Optional[str]:
    r = view.row(pid)
    if not r:
        return None
    st = _st(view, pid)
    if _f(st.get("gp")) < 5:
        return None
    name = r.get("name")
    if r.get("pos") == "G":
        sa = _f(st.get("shots_against") or st.get("goalie_shots_against"))
        ga = _f(st.get("goalie_ga") or st.get("ga"))
        xga = _f(st.get("goalie_xga"))
        if sa < 50 or xga <= 0:
            return None
        gsax = xga - ga
        verdict = "playing above his chances" if gsax > 3 else ("costing goals on the shots he faces" if gsax < -3 else "about what the shots warranted")
        return f"Analytics check — {name}: {gsax:+.1f} goals saved above expected on {int(sa)} shots ({xga:.1f} xGA vs {int(ga)} allowed), {verdict}."
    bits: List[str] = []
    g, ixg = _f(st.get("g")), _f(st.get("ixg"))
    if ixg > 1:
        diff = g - ixg
        tag = "finishing well above his chances — likely to cool" if diff > 4 else ("snakebitten — goals should come" if diff < -4 else "finishing about as expected")
        bits.append(f"{int(g)} G on {ixg:.1f} ixG ({diff:+.1f}, {tag})")
    xf, xa = _f(st.get("xgf")), _f(st.get("xga"))
    if xf + xa > 3:
        share = xf / (xf + xa) * 100
        bits.append(f"{share:.1f}% xGF on ice ({'drives play' if share >= 53 else ('gets outplayed' if share <= 47 else 'even')})")
    cf, ca = _f(st.get("cf")), _f(st.get("ca"))
    if cf + ca > 20:
        bits.append(f"{cf / (cf + ca) * 100:.1f}% CF")
    gf, ga = _f(st.get("gf_on")), _f(st.get("ga_on"))
    if gf + ga >= 6 and xf + xa > 3:
        luck = (gf - xf) - (ga - xa)
        if abs(luck) >= 3:
            bits.append(f"on-ice results {luck:+.1f} goals vs expected ({'riding luck' if luck > 0 else 'unlucky'})")
    return f"Analytics check — {name}: " + "; ".join(bits) + "." if bits else None


def team_line(view: LiveView, tid: str) -> Optional[str]:
    xf = xa = gf = ga = 0.0
    sk_g = sk_s = 0.0
    for r in view.team_rows(tid):
        st = _st(view, r["id"])
        if r.get("pos") == "G":
            continue
        toi = _f(st.get("toi_sec"))
        w = toi / 3600.0 if toi > 0 else 0.0
        xf += _f(st.get("xgf")) * w
        xa += _f(st.get("xga")) * w
        sk_g += _f(st.get("g"))
        sk_s += _f(st.get("sog"))
    if xf + xa <= 0:
        return None
    share = xf / (xf + xa) * 100
    goalie = None
    for r in view.team_rows(tid):
        if r.get("pos") == "G":
            st = _st(view, r["id"])
            if goalie is None or _f(st.get("gp")) > _f(_st(view, goalie["id"]).get("gp")):
                goalie = r
    parts = [f"{share:.1f}% of expected goals at 5v5-weighted ice time ({'controlling play' if share >= 52 else ('being outchanced' if share <= 48 else 'even play')})"]
    if goalie:
        gst = _st(view, goalie["id"])
        xga_g, ga_g = _f(gst.get("goalie_xga")), _f(gst.get("goalie_ga") or gst.get("ga"))
        if xga_g > 0:
            parts.append(f"starter {goalie['name']} {xga_g - ga_g:+.1f} GSAx")
        sa = _f(gst.get("shots_against") or gst.get("goalie_shots_against"))
        if sa > 0 and sk_s > 0:
            pdo = (sk_g / sk_s + (1 - ga_g / sa)) * 100
            parts.append(f"PDO {pdo:.1f} ({'running hot' if pdo > 101.5 else ('running cold' if pdo < 98.5 else 'normal')})")
    return "Analytics check — " + str((view.team_meta.get(tid) or {}).get("name") or tid) + ": " + "; ".join(parts) + "."


_TEAM_INTENTS = ("team_plan", "diagnose", "team_card", "standings", "streak", "luck", "strategy", "teamq", "reason:interpret",
                 "reason:sustain", "reason:pattern", "reason:trend", "reason:overview", "reason:strengths", "reason:confidence",
                 "reason:predict", "reason:playoff_race", "verdict:blame")


def attach(view: LiveView, out: Dict[str, Any], intent: str, q: str, slots: Dict[str, Any]) -> None:
    """Append up to two analytics lines that test the answer."""
    if not isinstance(out, dict) or out.get("_evidence_done"):
        return
    lines: List[str] = []
    try:
        pids: List[str] = list(out.get("focus_ids") or [])
        if not pids:
            pids = [c[0] for c in (slots.get("players") or []) if c][:2]
        for pid in pids[:2]:
            if pid in view.players:
                ln = player_line(view, pid)
                if ln:
                    lines.append(ln)
        team_q = any(str(intent).startswith(k) for k in _TEAM_INTENTS) or bool(re.search(r"\b(we|us|our|team|club)\b", q))
        if team_q and len(lines) < 2:
            tids = [str(t) for t in (slots.get("teams") or [])]
            tid = view.user_tid if (re.search(r"\b(we|us|our|my)\b", q) or not tids) else tids[0]
            ln = team_line(view, tid)
            if ln:
                lines.append(ln)
    except Exception:
        return
    if lines:
        out.setdefault("analysis", []).extend(lines[:2])
    out["_evidence_done"] = True
