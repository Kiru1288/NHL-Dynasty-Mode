"""
Salary-cap growth model (F11).

The upper limit used to grow by one revenue formula (~4%/yr with small nudges), so
every simulated summer looked the same. This model builds next season's growth from
~30 named factors read from the live league (revenue, losing clubs, escrow, markets,
playoff runs, superstars, the Cup winner, the national TV deal, the Board's rules,
the economy...), each with its own signed contribution, so the cap can jump, stall
or even dip, and the UI can show *why*.

Every 10 simulated seasons the national TV deal expires and the user sits at the
table (``tv_deal_options`` / ``choose_tv_deal``). The deal they sign shapes the
TV-related factors for the next decade.

Randomness (economy, currency, one-off events) is seeded per save + season, so a
season's explanation never changes between reads.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional

TV_DEAL_TERM = 10
GROWTH_MIN = -0.035
GROWTH_MAX = 0.10

#: User choices at a TV negotiation (applied for the next decade).
TV_DEAL_OPTIONS: Dict[str, Dict[str, Any]] = {
    "long_stable": {
        "label": "Long, stable national deal",
        "blurb": "Ten years with a fixed escalator. Steady, predictable cap growth; little upside.",
        "signing_bump": 0.010, "escalator": 0.009, "volatility": 0.002, "small_market": 0.000,
    },
    "max_upfront": {
        "label": "Max money up front",
        "blurb": "A big signing jump in year one, then thin escalators. Great now, tight later.",
        "signing_bump": 0.040, "escalator": 0.002, "volatility": 0.003, "small_market": 0.000,
    },
    "streaming_gamble": {
        "label": "Streaming revenue share",
        "blurb": "Smaller guarantee plus a share of streaming growth. Higher expected growth, big yearly swings.",
        "signing_bump": 0.000, "escalator": 0.012, "volatility": 0.012, "small_market": 0.000,
    },
    "regional_rights": {
        "label": "Clubs keep regional rights",
        "blurb": "Big markets sell their own games. League money grows, but small markets fall behind.",
        "signing_bump": 0.008, "escalator": 0.010, "volatility": 0.005, "small_market": -0.004,
    },
}
_LEGACY_DEAL = {"key": "legacy", "label": "Current national deal", "signing_bump": 0.0,
                "escalator": 0.006, "volatility": 0.003, "small_market": 0.0}


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(d)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _rng(session: Any, year: int, tag: str) -> random.Random:
    sid = str(getattr(session, "session_id", "") or getattr(session, "seed", "") or "save")
    h = hashlib.sha256(f"capgrowth|{sid}|{year}|{tag}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _league(session: Any) -> Any:
    return getattr(getattr(session, "sim", None), "league", None)


def _season(session: Any) -> int:
    return int(_f(getattr(session, "season_calendar_year", 2025), 2025))


# ---------------------------------------------------------------------------
# TV deal state
# ---------------------------------------------------------------------------

def tv_deal_state(session: Any) -> Dict[str, Any]:
    st = getattr(session, "tv_deal", None)
    if not isinstance(st, dict) or "start_year" not in st:
        st = dict(_LEGACY_DEAL)
        st.update({"start_year": _season(session), "term": TV_DEAL_TERM, "history": []})
        try:
            session.tv_deal = st
        except Exception:
            pass
    return st


def tv_deal_year(session: Any, season_next: Optional[int] = None) -> int:
    """1-based year of the deal for the season that starts next."""
    st = tv_deal_state(session)
    nxt = int(season_next if season_next is not None else _season(session) + 1)
    return max(1, nxt - int(st.get("start_year", nxt)) + 1)


def tv_negotiation_open(session: Any) -> bool:
    """The deal runs out: the season after next would be year 11."""
    st = tv_deal_state(session)
    return tv_deal_year(session) > int(st.get("term", TV_DEAL_TERM))


def tv_deal_options(session: Any) -> Dict[str, Any]:
    st = tv_deal_state(session)
    return {
        "open": tv_negotiation_open(session),
        "current": {k: st.get(k) for k in ("key", "label", "start_year", "term")},
        "deal_year": tv_deal_year(session),
        "options": [dict(v, key=k) for k, v in TV_DEAL_OPTIONS.items()],
        "history": list(st.get("history") or [])[-5:],
    }


def choose_tv_deal(session: Any, key: str, *, auto: bool = False) -> Dict[str, Any]:
    if key not in TV_DEAL_OPTIONS:
        raise ValueError(f"unknown TV deal option: {key}")
    st = tv_deal_state(session)
    if not tv_negotiation_open(session):
        raise ValueError("No national TV negotiation is open.")
    opt = TV_DEAL_OPTIONS[key]
    hist = list(st.get("history") or [])
    hist.append({"key": st.get("key"), "label": st.get("label"), "start_year": st.get("start_year")})
    new = {"key": key, "label": opt["label"], "start_year": _season(session) + 1, "term": TV_DEAL_TERM,
           "signing_bump": opt["signing_bump"], "escalator": opt["escalator"], "volatility": opt["volatility"],
           "small_market": opt["small_market"], "history": hist[-10:], "auto_signed": bool(auto)}
    session.tv_deal = new
    return new


# ---------------------------------------------------------------------------
# Inputs from the live league
# ---------------------------------------------------------------------------

def _teams(session: Any) -> List[Any]:
    lg = _league(session)
    return list(getattr(lg, "teams", None) or [])


def _team_ovr(p: Any) -> float:
    for k in ("overall", "ovr"):
        v = getattr(p, k, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if v is not None:
            x = _f(v, 0)
            return x * 99.0 if 0 < x <= 1.5 else x
    return 0.0


def _league_gpg(session: Any) -> Optional[float]:
    games = list(getattr(session, "game_results", None) or [])[-600:]
    tot = n = 0
    for g in games:
        if not isinstance(g, dict):
            continue
        try:
            tot += int(g.get("home_goals", 0) or 0) + int(g.get("away_goals", 0) or 0)
            n += 1
        except (TypeError, ValueError):
            continue
    return (tot / (2.0 * n)) if n >= 100 else None


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _factor(rows: List[Dict[str, Any]], key: str, label: str, impact: float, detail: str, group: str) -> None:
    rows.append({
        "key": key, "label": label, "impact_pct": round(impact * 100.0, 2), "detail": detail, "group": group,
        "direction": "up" if impact > 0.0005 else ("down" if impact < -0.0005 else "flat"),
    })


def compute_cap_growth(
    session: Any,
    league_state: Optional[Dict[str, Any]] = None,
    team_rows: Optional[List[Dict[str, Any]]] = None,
    *,
    gov_growth_pts: float = 0.0,
) -> Dict[str, Any]:
    """Growth fraction for next season with a factor-by-factor explanation."""
    ls = league_state or {}
    rows: List[Dict[str, Any]] = []
    year_next = _season(session) + 1
    tr = [r for r in (team_rows or []) if isinstance(r, dict)]
    n_teams = max(1, len(tr) or len(_teams(session)) or 32)
    r_econ = _rng(session, year_next, "economy")
    r_evt = _rng(session, year_next, "events")

    # --- Revenue core ----------------------------------------------------------
    _factor(rows, "hrr_trend", "Hockey-related revenue trend", 0.020,
            "Baseline growth of league revenue (gate, sponsorship, media).", "revenue")
    health = _f(ls.get("revenue_health"), 0.6)
    _factor(rows, "revenue_health", "League revenue health", _clamp((health - 0.70) * 0.05, -0.015, 0.015),
            f"Revenue health index {health:.2f} (0.70 = normal).", "revenue")
    losing = int(_f(ls.get("losing_teams_count"), 0))
    _factor(rows, "losing_clubs", "Clubs losing money", -max(0, losing - 4) * 0.0018 + (0.003 if losing == 0 else 0.0),
            f"{losing} clubs lost money last season" + (" — every club profitable." if losing == 0 else "."), "finance")
    sm_be = _f(ls.get("small_market_break_even"), 0.7)
    _factor(rows, "small_market_health", "Small-market break-even", (sm_be - 0.70) * 0.02,
            f"{sm_be * 100:.0f}% of small-market clubs broke even.", "finance")
    escrow = _f(ls.get("escrow_progress"), 1.0)
    _factor(rows, "escrow", "Escrow balance", _clamp(escrow - 1.0, -0.15, 0.15) * 0.06,
            "Players owed escrow back: revenue came in under projections." if escrow < 0.98 else
            ("Escrow over-collected: revenue beat projections." if escrow > 1.02 else "Escrow close to target."), "finance")
    cba = _f(ls.get("cba_pressure"), 0.3)
    _factor(rows, "labor", "Labor climate (CBA pressure)", -max(0.0, cba - 0.55) * 0.05 + (0.002 if cba < 0.30 else 0.0),
            f"CBA pressure {cba:.2f}" + (" — owners push to slow the cap." if cba >= 0.55 else " — labor peace."), "labor")
    sstar = _f(ls.get("superstar_revenue_share"), 0.0)
    _factor(rows, "superstar_revenue", "Superstar revenue boost", min(0.008, sstar * 0.15),
            f"Stars drive {sstar * 100:.1f}% of league revenue.", "revenue")

    # --- Markets and on-ice story ---------------------------------------------
    teams = _teams(session)
    elite = 0
    for t in teams:
        for p in list(getattr(t, "roster", None) or []):
            if _team_ovr(p) >= 92.0:
                elite += 1
    _factor(rows, "superstars", "Superstar count", _clamp((elite - 6) * 0.0012, -0.006, 0.008),
            f"{elite} players rated 92+ (league draws, jersey sales).", "on_ice")
    tier_by_tid = {str(r.get("id") or r.get("team_id") or ""): str(r.get("market_tier") or "") for r in tr}
    pts = sorted(((str(getattr(t, "team_id", "")), _f(getattr(t, "point_pct", 0.5), 0.5)) for t in teams),
                 key=lambda kv: -kv[1])
    playoff_ids = [tid for tid, _ in pts[:16]]
    big_in = sum(1 for tid in playoff_ids if tier_by_tid.get(tid) == "Large")
    big_total = max(1, sum(1 for v in tier_by_tid.values() if v == "Large"))
    _factor(rows, "big_market_playoffs", "Big markets in the playoffs", (big_in / big_total - 0.5) * 0.012,
            f"{big_in} of {big_total} large-market clubs made the playoffs.", "on_ice")
    champ = str(getattr(session, "stanley_cup_winner", "") or getattr(session, "champion_id", "") or "")
    ct = tier_by_tid.get(champ, "")
    _factor(rows, "cup_market", "Stanley Cup winner's market",
            {"Large": 0.004, "Mid": 0.001, "Small": -0.0015}.get(ct, 0.0),
            f"Champion's market: {ct or 'unknown'} (Final ratings and merchandise).", "on_ice")
    fans = [_f(r.get("fan_sentiment"), 55) for r in tr] or [55.0]
    avg_fan = sum(fans) / len(fans)
    _factor(rows, "attendance", "Attendance and fan sentiment", (avg_fan - 58.0) * 0.0006,
            f"League fan sentiment {avg_fan:.0f}/100.", "revenue")
    gpg = _league_gpg(session)
    if gpg is not None:
        _factor(rows, "entertainment", "Scoring / entertainment", _clamp((gpg - 3.0) * 0.008, -0.004, 0.004),
                f"{gpg:.2f} goals per team-game.", "on_ice")
    pcts = [v for _, v in pts] or [0.5]
    mu = sum(pcts) / len(pcts)
    sd = math.sqrt(sum((x - mu) ** 2 for x in pcts) / len(pcts))
    _factor(rows, "parity", "Competitive balance", _clamp((0.075 - sd) * 0.05, -0.003, 0.003),
            "Tight standings keep more markets engaged." if sd < 0.075 else "Lopsided standings cool weaker markets.",
            "on_ice")
    rev = [_f(r.get("revenue"), 0) for r in tr]
    if rev:
        top = sorted(rev, reverse=True)
        share = sum(top[:5]) / max(1.0, sum(top))
        _factor(rows, "revenue_concentration", "Revenue concentration", -max(0.0, share - 0.22) * 0.03,
                f"Top 5 clubs earn {share * 100:.0f}% of league revenue.", "finance")

    # --- National TV deal ------------------------------------------------------
    st = tv_deal_state(session)
    dy = tv_deal_year(session, year_next)
    term = int(st.get("term", TV_DEAL_TERM))
    pending_tv = dy > term
    if pending_tv:
        # Deal expired and the user has not signed yet: project on a stopgap rollover.
        # (Auto-signed at the cap rollover if the user never chooses.)
        _factor(rows, "tv_expired", "National TV deal expired", -0.004,
                "No new national deal signed yet - negotiate it in the Cap Ledger.", "media")
    esc = _f(st.get("escalator"), 0.006) * (0.5 if pending_tv else 1.0)
    phase_mult = 1.15 if dy <= 3 else (1.0 if dy <= 7 else 0.75)
    _factor(rows, "tv_escalator", "National TV deal escalator", esc * phase_mult,
            f"{st.get('label', 'TV deal')}, year {dy} of {term}.", "media")
    if dy == 1 and _f(st.get("signing_bump"), 0) > 0:
        _factor(rows, "tv_signing", "New TV deal signing money", _f(st.get("signing_bump"), 0),
                "First season of the new national deal.", "media")
    vol = _f(st.get("volatility"), 0.003)
    if vol > 0:
        swing = r_econ.gauss(0.0, vol)
        _factor(rows, "tv_variance", "Media revenue swing", swing,
                "Streaming/ratings came in above plan." if swing > 0 else "Ratings softer than plan.", "media")
    smd = _f(st.get("small_market"), 0.0)
    if smd:
        _factor(rows, "regional_rights", "Regional-rights gap", smd,
                "Small markets fall behind under the regional-rights deal.", "media")
    if dy >= term - 1:
        _factor(rows, "tv_uncertainty", "TV deal expiring soon", -0.003,
                "Owners budget cautiously with the national deal up for renewal.", "media")

    # --- Economy and one-off events (seeded per season) --------------------------
    macro = r_econ.gauss(0.0, 0.006)
    if r_econ.random() < 0.07:
        macro -= r_econ.uniform(0.015, 0.035)
        macro_txt = "Economic downturn: sponsors and season-ticket renewals fell."
    elif r_econ.random() < 0.08:
        macro += r_econ.uniform(0.006, 0.015)
        macro_txt = "Strong economy: corporate spending and premium seating up."
    else:
        macro_txt = "Normal economy."
    _factor(rows, "economy", "Economy", macro, macro_txt, "economy")
    fx = r_econ.gauss(0.0, 0.0025)
    _factor(rows, "currency", "Canadian dollar", fx, "CAD strengthened vs USD." if fx > 0 else "CAD weakened vs USD.",
            "economy")
    infl = r_econ.uniform(0.015, 0.035)
    _factor(rows, "inflation", "Inflation pass-through", (infl - 0.025) * 0.3,
            f"Inflation {infl * 100:.1f}%.", "economy")
    _factor(rows, "digital", "Digital / sponsorship growth", 0.003 + r_evt.uniform(-0.001, 0.002),
            "Jersey patches, in-arena tech and digital sponsorship.", "revenue")
    if r_evt.random() < 0.15:
        _factor(rows, "arena", "New arena openings", r_evt.uniform(0.002, 0.005),
                "A new building opened with higher premium-seat revenue.", "revenue")
    if r_evt.random() < 0.12:
        _factor(rows, "international", "International showcase", r_evt.uniform(0.001, 0.004),
                "Global Series / best-on-best event lifted international revenue.", "revenue")
    if r_evt.random() < 0.06:
        _factor(rows, "work_stoppage_threat", "Work-stoppage talk", -r_evt.uniform(0.004, 0.010),
                "Labor posturing scared off some sponsors.", "labor")
    prev_n = int(getattr(session, "_cap_model_team_count", 0) or 0)
    if prev_n and n_teams > prev_n:
        _factor(rows, "expansion", "Expansion fee and new market", 0.008 * (n_teams - prev_n),
                f"{n_teams - prev_n} expansion club(s) added revenue.", "revenue")
    try:
        session._cap_model_team_count = n_teams
    except Exception:
        pass

    # --- Governance / smoothing ---------------------------------------------------
    if gov_growth_pts:
        _factor(rows, "board_rules", "Board of Governors cap rules", gov_growth_pts / 100.0,
                "Rules passed by the Board on cap growth.", "governance")
    hist = list(getattr(_league(session), "cap_history", None) or [])
    last_pct = _f((hist[-1] or {}).get("cap_change_percent"), 4.0) / 100.0 if hist and isinstance(hist[-1], dict) else 0.04
    raw = sum(r["impact_pct"] for r in rows) / 100.0
    if last_pct > 0.07 and raw > 0.05:
        smooth = -(raw - 0.05) * 0.4
        _factor(rows, "smoothing", "Cap smoothing", smooth, "After last year's big jump, part of the rise is deferred.",
                "governance")
    elif last_pct < 0.0 and raw < 0.02:
        _factor(rows, "smoothing", "Cap smoothing", 0.006, "League cushions a second weak year.", "governance")
    growth = _clamp(sum(r["impact_pct"] for r in rows) / 100.0, GROWTH_MIN, GROWTH_MAX)
    ups = sorted((r for r in rows if r["direction"] == "up" and r["key"] != "hrr_trend"), key=lambda r: -r["impact_pct"])
    downs = sorted((r for r in rows if r["direction"] == "down"), key=lambda r: r["impact_pct"])
    if growth < 0:
        headline = "The cap goes DOWN: " + ", ".join(r["label"].lower() for r in downs[:2]) + "."
    elif growth < 0.02:
        headline = "The cap barely moves: " + ", ".join(r["label"].lower() for r in downs[:2]) + " offset revenue growth."
    elif growth >= 0.065:
        headline = "Big cap jump: " + ", ".join(r["label"].lower() for r in ups[:2]) + "."
    else:
        lead = ups[:1] + downs[:1]
        headline = "Normal rise; " + " vs ".join(r["label"].lower() for r in lead) + " were the swing factors."
    return {
        "growth": round(growth, 4),
        "growth_pct": round(growth * 100.0, 2),
        "season_next": year_next,
        "factors": sorted(rows, key=lambda r: -abs(r["impact_pct"])),
        "headline": headline,
        "tv_deal": {"label": st.get("label"), "key": st.get("key"), "deal_year": dy, "term": term},
        "tv_negotiation_open": tv_negotiation_open(session),
    }


def record_cap_explanation(session: Any, model: Dict[str, Any], cap_row: Dict[str, Any]) -> None:
    """Keep the explanation for the cap actually set at rollover (Cap Ledger history).
    An expired TV deal nobody signed is auto-renewed on the long, stable deal here."""
    if tv_negotiation_open(session):
        try:
            choose_tv_deal(session, "long_stable", auto=True)
        except Exception:
            pass
    hist = list(getattr(session, "cap_growth_history", None) or [])
    entry = {
        "season": int(model.get("season_next") or 0),
        "previous_cap": cap_row.get("previous_cap"),
        "new_cap": cap_row.get("upperLimit", cap_row.get("current_cap")),
        "change_pct": cap_row.get("cap_change_percent"),
        "model_growth_pct": model.get("growth_pct"),
        "headline": model.get("headline"),
        "factors": list(model.get("factors") or [])[:12],
        "tv_deal": model.get("tv_deal"),
    }
    hist = [h for h in hist if int(h.get("season") or 0) != entry["season"]]
    hist.append(entry)
    session.cap_growth_history = hist[-15:]
