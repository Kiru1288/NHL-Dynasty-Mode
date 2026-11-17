"""Franchise Command: commissioner sliders that reshape the save live.

Fifty sliders in five groups. Each one goes through one of three channels, so a
change shows up on the very next sim tick:

* ``gov``  – added on top of the Board of Governors modifiers. Everything that
  already reads ``league.governance_modifiers`` (game rules, injuries, trades,
  revenue, cap rules, scouting...) reacts without new wiring.
* ``abs``  – an absolute on-ice format value (OT length, ties, points). Only
  written when it differs from the NHL default, so it never hides a voted rule
  when left alone.
* ``knob`` – a runtime multiplier in ``app.sim_engine.tuning.command_knobs``
  read by progression, the death/disaster/plot/tension rolls, the engine's
  chaos and morale nudges, and contract asks.

Progression is judged against the scoring environment the sliders create
(``scoring_env``): a 70-point defenceman in a high-scoring league is measured
against a higher bar than the same 70 points in a dead-puck league.

Also here: saving a franchise to disk, listing saves and loading one back, and
carrying the sliders into a brand-new sim.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

GROUPS: List[Tuple[str, str, str]] = [
    ("ice", "On the Ice", "How games are played and scored."),
    ("growth", "Player Growth", "How ratings rise and fall."),
    ("chaos", "League Life", "Drama, luck and wild cards."),
    ("money", "Money & Cap", "Salary cap, contracts and revenue."),
    ("world", "Front Office World", "Markets, movement and the map."),
]


def S(sid, group, label, desc, lo, hi, step, default, unit, channel, key, *, low="", high="", fmt=""):
    return {
        "id": sid, "group": group, "label": label, "desc": desc, "min": lo, "max": hi, "step": step,
        "default": default, "unit": unit, "channel": channel, "key": key, "low": low, "high": high, "fmt": fmt,
    }


SLIDERS: List[Dict[str, Any]] = [
    # ── On the ice ─────────────────────────────────────────────────────────
    S("scoring_pace", "ice", "Scoring Pace", "Goals per game across the league.", -18, 25, 1, 0, "%", "gov", "scoring_pace", low="Dead puck", high="Run & gun"),
    S("penalty_rate", "ice", "Ref Whistle", "How often penalties get called.", -35, 45, 5, 0, "%", "gov", "penalty_rate", low="Let 'em play", high="Every hook"),
    S("pp_efficiency", "ice", "Power Play Punch", "How dangerous a man advantage is.", -25, 35, 5, 0, "%", "gov", "pp_efficiency", low="Toothless", high="Lethal"),
    S("home_ice", "ice", "Home Ice", "How much the home crowd helps.", -60, 80, 10, 0, "%", "gov", "home_ice", low="Neutral site", high="Fortress"),
    S("injury_rate", "ice", "Injury Rate", "How often players get hurt.", -35, 25, 5, 0, "%", "gov", "injury_rate", low="Bubble wrap", high="Bone crusher"),
    S("ot_minutes", "ice", "Overtime Length", "Minutes of 3-on-3 before a shootout.", 3, 20, 1, 5, " min", "abs", "ot_minutes", low="Quick", high="Marathon"),
    S("ties_allowed", "ice", "Ties Allowed", "Games can end tied after overtime.", 0, 1, 1, 0, "", "abs", "ties_allowed", low="Off", high="On", fmt="toggle"),
    S("pts_reg_win", "ice", "Points: Regulation Win", "Standings points for a 60-minute win.", 2, 4, 1, 2, " pts", "abs", "pts_reg_win"),
    S("pts_ot_win", "ice", "Points: OT Win", "Standings points for an overtime or shootout win.", 1, 4, 1, 2, " pts", "abs", "pts_ot_win"),
    S("pts_otl", "ice", "Loser Point", "Standings points for losing in overtime.", 0, 2, 1, 1, " pts", "abs", "pts_otl"),
    # ── Player growth ──────────────────────────────────────────────────────
    S("prog_growth", "growth", "Growth Speed", "How fast good seasons raise overall.", 40, 200, 10, 100, "%", "knob", "prog_growth", low="Slow burn", high="Rocket"),
    S("prog_decline", "growth", "Slump Penalty", "How hard bad seasons drop overall.", 40, 200, 10, 100, "%", "knob", "prog_decline", low="Forgiving", high="Brutal"),
    S("aging_curve", "growth", "Aging Curve", "How hard veterans (31+) decline.", 40, 200, 10, 100, "%", "knob", "aging_curve", low="Ageless", high="Cliff"),
    S("prospect_growth", "growth", "Prospect Growth", "Junior and AHL development speed.", 40, 200, 10, 100, "%", "knob", "prospect_growth", low="Raw", high="Fast track"),
    S("goalie_growth", "growth", "Goalie Growth", "How fast goalies develop.", 40, 200, 10, 100, "%", "knob", "goalie_growth", low="Late bloomers", high="Instant"),
    S("production_weight", "growth", "Points Matter", "How much scoring above or below expectation moves overall.", 0, 200, 10, 100, "%", "knob", "production_weight", low="Ignore box score", high="Box score is king"),
    S("defense_point_credit", "growth", "D-Man Points Credit", "How much a defenceman's points count toward his rating.", 25, 200, 25, 100, "%", "knob", "defense_point_credit", low="Shutdown first", high="Offence first"),
    S("era_normalize", "growth", "Era-Adjust Production", "Judge points against the league's scoring level, so a high-scoring league does not inflate ratings.", 0, 1, 1, 1, "", "knob", "era_normalize", low="Raw points", high="Era-adjusted", fmt="toggle"),
    S("offseason_growth", "growth", "Summer Work", "Extra overall a good season adds over the summer.", 0, 200, 25, 100, "%", "knob", "offseason_growth", low="None", high="Double"),
    S("wjc_impact", "growth", "World Juniors Impact", "How much the WJC moves draft stock.", 0, 250, 25, 100, "%", "knob", "wjc_impact", low="Ignored", high="Makes careers"),
    # ── League life ────────────────────────────────────────────────────────
    S("chaos", "chaos", "Upset Factor", "Game-to-game randomness and upsets.", 40, 220, 10, 100, "%", "knob", "chaos", low="Chalk", high="Anything goes"),
    S("morale_volatility", "chaos", "Mood Swings", "How fast locker-room morale swings.", 40, 220, 10, 100, "%", "knob", "morale_volatility", low="Steady", high="Soap opera"),
    S("trade_volume", "chaos", "Trade Frenzy", "How often clubs make trades.", -50, 70, 10, 0, "%", "gov", "trade_volume", low="Frozen", high="Wheel & deal"),
    S("trade_demand_rate", "chaos", "Trade Requests", "How often unhappy players ask out.", -50, 50, 10, 0, "%", "gov", "trade_demand_rate", low="Loyal", high="Divas"),
    S("political_tension", "chaos", "International Tension", "Chance a tournament turns political.", 0, 300, 25, 100, "%", "knob", "political_tension", low="Peaceful", high="Cold war"),
    S("death_rate", "chaos", "Tragedy Odds", "Chance of a player death in a season.", 0, 300, 25, 100, "%", "knob", "death_rate", low="Never", high="Grim"),
    S("disaster_rate", "chaos", "Disaster Odds", "Chance of a team disaster and emergency draft.", 0, 500, 50, 100, "%", "knob", "disaster_rate", low="Never", high="Cursed"),
    S("plot_heat", "chaos", "Investigator Heat", "How likely a cap-dump plot gets caught.", 50, 200, 10, 100, "%", "knob", "plot_heat", low="Sleepy cops", high="Sherlock"),
    S("goofy_events", "chaos", "Goofy Events", "Mystery boxes, wild promos and silly moments.", 0, 1, 1, 1, "", "knob", "goofy_events", low="Off", high="On", fmt="toggle"),
    S("scouting_budget", "chaos", "Scouting Budget", "League-wide scouting spend and accuracy.", -35, 45, 5, 0, "%", "gov", "scouting_budget", low="Shoestring", high="Deep pockets"),
    # ── Money & cap ────────────────────────────────────────────────────────
    S("cap_growth", "money", "Cap Growth", "Extra yearly salary-cap growth.", -1.5, 1.5, 0.25, 0, " pts", "gov", "cap_growth", low="Flat cap", high="Boom"),
    S("floor_ratio", "money", "Cap Floor", "Shift the payroll floor as a share of the cap.", -0.06, 0.05, 0.01, 0, "", "gov", "floor_ratio", low="Low floor", high="High floor", fmt="pct"),
    S("max_salary_pct", "money", "Max Contract", "Shift the max salary share of the cap.", -0.05, 0.05, 0.01, 0, "", "gov", "max_salary_pct", low="Capped stars", high="Mega deals", fmt="pct"),
    S("min_salary_m", "money", "League Minimum", "Raise or lower the minimum salary.", -0.1, 0.3, 0.05, 0, "M", "gov", "min_salary_m", low="Cheap", high="Raise", fmt="money"),
    S("bonus_pct", "money", "Bonus Room", "How much of a deal can be bonuses.", -0.15, 0.15, 0.05, 0, "", "gov", "bonus_pct", low="Strict", high="Loose", fmt="pct"),
    S("fa_bonus_demand", "money", "Free-Agent Bonus Ask", "How hard free agents push for bonuses.", -25, 25, 5, 0, "%", "gov", "fa_bonus_demand", low="Humble", high="Greedy"),
    S("contract_asks", "money", "Contract Asks", "Every player's salary ask, up or down.", 70, 140, 5, 100, "%", "knob", "contract_asks", low="Bargains", high="Highway robbery"),
    S("retention_slots", "money", "Retention Slots", "Retained-salary slots per club.", -2, 2, 1, 0, "", "gov", "retention_slots", low="Fewer", high="More"),
    S("retention_max_pct", "money", "Max Retention", "How much salary can be retained in trades.", -20, 20, 5, 0, "%", "gov", "retention_max_pct", low="Less", high="More"),
    S("rev_all", "money", "League Revenue", "Revenue for every club.", -12, 15, 1, 0, "%", "gov", "rev_all", low="Recession", high="Gold rush"),
    S("revenue_share", "money", "Revenue Sharing", "How much rich clubs share with poor ones.", -4, 6, 1, 0, "%", "gov", "revenue_share", low="Every club alone", high="Socialist"),
    S("opex", "money", "Operating Costs", "Cost of running a club.", -4.5, 6, 0.5, 0, "%", "gov", "opex", low="Lean", high="Bloated"),
    S("star_rev", "money", "Star Power", "Extra revenue stars bring in.", -20, 40, 5, 0, "%", "gov", "star_rev", low="Team first", high="Hollywood"),
    S("playoff_rev", "money", "Playoff Gate", "Revenue from home playoff games.", -40, 50, 10, 0, "%", "gov", "playoff_rev", low="Modest", high="Jackpot"),
    S("fan", "money", "Fan Patience", "Fan sentiment baseline.", -25, 25, 5, 0, "", "gov", "fan", low="Boo birds", high="Diehards"),
    # ── Front office world ────────────────────────────────────────────────
    S("value_growth", "world", "Franchise Values", "How fast franchise values grow.", -2, 2, 0.5, 0, " pts", "gov", "value_growth", low="Flat", high="Bubble"),
    S("relocation_ease", "world", "Relocation Risk", "How easily struggling clubs move.", -0.15, 0.15, 0.05, 0, "", "gov", "relocation_ease", low="Rooted", high="Moving vans", fmt="pct"),
    S("expansion_pressure", "world", "Expansion Pressure", "How hard new cities push for teams.", -40, 50, 10, 0, "%", "gov", "expansion_pressure", low="Closed", high="Gold rush"),
    S("rev_small", "world", "Small-Market Boost", "Revenue for small-market clubs.", -9, 12, 1, 0, "%", "gov", "rev_small", low="Squeezed", high="Lifted"),
    S("rev_large", "world", "Big-Market Boost", "Revenue for big-market clubs.", -9, 9, 1, 0, "%", "gov", "rev_large", low="Levelled", high="Empires"),
]

SLIDER_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in SLIDERS}
assert len(SLIDERS) == 50, len(SLIDERS)

PRESETS: List[Dict[str, Any]] = [
    {"id": "standard", "label": "NHL Standard", "blurb": "Every slider at the default.", "values": {}},
    {"id": "run_gun", "label": "Run & Gun", "blurb": "Goals everywhere, power plays that kill, and era-adjusted ratings so nobody balloons.",
     "values": {"scoring_pace": 25, "pp_efficiency": 30, "penalty_rate": 25, "ot_minutes": 7, "chaos": 130, "era_normalize": 1}},
    {"id": "dead_puck", "label": "Dead Puck", "blurb": "Trap hockey, few calls, and every point a big deal.",
     "values": {"scoring_pace": -18, "penalty_rate": -30, "pp_efficiency": -15, "chaos": 70, "home_ice": 20}},
    {"id": "chaos", "label": "Chaos League", "blurb": "Upsets, mood swings, trade frenzy and wild cards.",
     "values": {"chaos": 220, "morale_volatility": 200, "trade_volume": 70, "trade_demand_rate": 50, "political_tension": 250,
                "disaster_rate": 300, "goofy_events": 1, "injury_rate": 15}},
    {"id": "fast_dynasty", "label": "Fast Dynasty", "blurb": "Kids develop fast, veterans fade fast, the league turns over.",
     "values": {"prog_growth": 160, "prospect_growth": 170, "aging_curve": 150, "goalie_growth": 150, "offseason_growth": 150}},
    {"id": "hardcore", "label": "Hardcore Realism", "blurb": "Slow growth, harsh slumps, tight money, no goofy stuff.",
     "values": {"prog_growth": 75, "prog_decline": 140, "aging_curve": 130, "goofy_events": 0, "cap_growth": -0.75,
                "contract_asks": 110, "plot_heat": 160, "pts_otl": 0, "ot_minutes": 5}},
    {"id": "big_money", "label": "Gold Rush", "blurb": "Booming revenue, rising cap, stars paid like movie stars.",
     "values": {"rev_all": 15, "cap_growth": 1.5, "star_rev": 40, "max_salary_pct": 0.05, "value_growth": 2, "playoff_rev": 50}},
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _snap(spec: Dict[str, Any], v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float(spec["default"])
    x = _clamp(x, float(spec["min"]), float(spec["max"]))
    step = float(spec["step"] or 1)
    x = round(round((x - float(spec["min"])) / step) * step + float(spec["min"]), 4)
    return _clamp(x, float(spec["min"]), float(spec["max"]))


def ensure_state(session: Any) -> Dict[str, Any]:
    st = getattr(session, "franchise_command", None)
    if not isinstance(st, dict):
        st = {"values": {}, "history": [], "preset": "standard"}
        try:
            session.franchise_command = st
        except Exception:
            _log.debug("franchise_command attach failed", exc_info=True)
    st.setdefault("values", {})
    st.setdefault("history", [])
    return st


def values(session: Any) -> Dict[str, float]:
    st = ensure_state(session)
    raw = st.get("values") or {}
    return {s["id"]: _snap(s, raw.get(s["id"], s["default"])) for s in SLIDERS}


def governance_overlay(session: Any) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(additive, absolute) modifiers the sliders add to the Board's."""
    add: Dict[str, float] = {}
    absolute: Dict[str, float] = {}
    st = getattr(session, "franchise_command", None)
    if not isinstance(st, dict):
        return add, absolute
    vals = values(session)
    for s in SLIDERS:
        v = vals[s["id"]]
        if s["channel"] == "gov" and abs(v) > 1e-9:
            add[s["key"]] = add.get(s["key"], 0.0) + v
        elif s["channel"] == "abs" and abs(v - float(s["default"])) > 1e-9:
            absolute[s["key"]] = v
    return add, absolute


def merge_overlay(session: Any, mods: Dict[str, float], limits: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Called by league_governance.sync_governance_to_league after the rulebook is summed."""
    add, absolute = governance_overlay(session)
    out = dict(mods)
    for k, v in add.items():
        lo, hi = limits.get(k, (-1e9, 1e9))
        spec = next((s for s in SLIDERS if s["key"] == k), None)
        lo2 = lo + (float(spec["min"]) if spec else 0.0)
        hi2 = hi + (float(spec["max"]) if spec else 0.0)
        out[k] = round(_clamp(float(out.get(k, 0.0)) + v, lo2, hi2), 4)
    for k, v in absolute.items():
        out[k] = v
    return out


def _league(session: Any) -> Any:
    sim = getattr(session, "sim", None)
    return getattr(sim, "league", None) if sim is not None else None


def runtime_knobs(session: Any) -> Dict[str, float]:
    vals = values(session)
    knobs: Dict[str, float] = {}
    for s in SLIDERS:
        if s["channel"] != "knob":
            continue
        v = vals[s["id"]]
        knobs[s["key"]] = v if s["fmt"] == "toggle" else v / 100.0
    # Scoring environment from the live game rules (sliders + voted rules together).
    env, d_env = 1.0, 1.0
    try:
        from app.sim_engine.league.game_rules import rules_for

        r = rules_for(_league(session))
        pp_vol = float(r["pp_mult"]) * float(r["penalty_mult"])
        env = float(r["scoring_mult"]) * (0.80 + 0.20 * pp_vol)
        # Defencemen quarterback the power play, so they gain a little more from it.
        d_env = 1.0 + 0.10 * (pp_vol - 1.0)
    except Exception:
        _log.debug("scoring env failed", exc_info=True)
    knobs["scoring_env"] = round(_clamp(env, 0.6, 1.6), 4)
    knobs["defense_scoring_env"] = round(_clamp(d_env, 0.8, 1.25), 4)
    return knobs


def apply_runtime(session: Any) -> Dict[str, float]:
    """Push this save's knobs into the engine. Cheap: run at the start of every sim day."""
    knobs = runtime_knobs(session)
    try:
        from app.sim_engine.tuning.command_knobs import set_knobs

        set_knobs(knobs)
    except Exception:
        _log.debug("set_knobs failed", exc_info=True)
    if isinstance(getattr(session, "franchise_command", None), dict):
        try:
            session.goofy_events_enabled = bool(knobs.get("goofy_events", 1.0) >= 0.5)
        except Exception:
            pass
    return knobs


def apply(session: Any) -> Dict[str, Any]:
    """Full apply: governance mirror (game rules, money, trades...) then runtime knobs."""
    try:
        from services.league_governance import sync_governance_to_league

        sync_governance_to_league(session)
    except Exception:
        _log.debug("governance sync failed", exc_info=True)
    return apply_runtime(session)


def set_values(session: Any, updates: Dict[str, Any], *, preset: str = "", replace: bool = False) -> Dict[str, Any]:
    st = ensure_state(session)
    cur = {} if replace else dict(st.get("values") or {})
    changed: List[str] = []
    for sid, v in (updates or {}).items():
        spec = SLIDER_BY_ID.get(str(sid))
        if spec is None:
            continue
        nv = _snap(spec, v)
        if abs(nv - float(cur.get(sid, spec["default"]))) > 1e-9:
            changed.append(spec["label"])
        cur[sid] = nv
    st["values"] = {k: v for k, v in cur.items() if abs(v - float(SLIDER_BY_ID[k]["default"])) > 1e-9}
    st["preset"] = preset or ("custom" if st["values"] else "standard")
    if changed or replace:
        hist = st.setdefault("history", [])
        hist.insert(0, {
            "date": str(getattr(session, "current_date", "") or getattr(session, "calendar_date", "") or ""),
            "season": getattr(session, "season_calendar_year", None),
            "what": (f"Preset: {preset}" if preset else ", ".join(changed[:4]) + (" +" + str(len(changed) - 4) if len(changed) > 4 else "")),
        })
        del hist[24:]
    apply(session)
    return payload(session)


def randomize(session: Any, intensity: float = 0.5) -> Dict[str, Any]:
    rng = random.Random(f"fc|{getattr(session, 'session_id', '')}|{time.time()}")
    intensity = _clamp(float(intensity), 0.1, 1.0)
    upd: Dict[str, float] = {}
    for s in SLIDERS:
        if s["fmt"] == "toggle" or s["channel"] == "abs":
            continue
        if rng.random() > 0.55:
            continue
        lo, hi, d = float(s["min"]), float(s["max"]), float(s["default"])
        span = (hi - d) if rng.random() < 0.5 else (lo - d)
        upd[s["id"]] = d + span * rng.uniform(0.25, 1.0) * intensity
    return set_values(session, upd, preset="Surprise Me", replace=True)


def environment_summary(session: Any) -> Dict[str, Any]:
    knobs = runtime_knobs(session)
    env = knobs["scoring_env"]
    d = knobs["defense_scoring_env"]
    adj = knobs.get("era_normalize", 1.0) >= 0.5
    pts70 = 70.0
    fwd_equiv = pts70 / env if adj else pts70
    d_equiv = pts70 / (env * d) if adj else pts70
    return {
        "scoring_env": env,
        "defense_env": round(env * d, 3),
        "era_adjusted": adj,
        "example": {
            "raw_points": 70,
            "forward_equivalent": round(fwd_equiv, 1),
            "defense_equivalent": round(d_equiv, 1),
        },
        "label": ("High-scoring league" if env >= 1.08 else "Low-scoring league" if env <= 0.93 else "Normal scoring"),
    }


def payload(session: Any) -> Dict[str, Any]:
    st = ensure_state(session)
    vals = values(session)
    league = _league(session)
    gov_rules: Dict[str, Any] = {}
    try:
        from app.sim_engine.league.game_rules import rules_for

        gov_rules = dict(rules_for(league))
    except Exception:
        pass
    changed = sum(1 for s in SLIDERS if abs(vals[s["id"]] - float(s["default"])) > 1e-9)
    return {
        "ok": True,
        "groups": [{"id": g, "label": l, "blurb": b} for g, l, b in GROUPS],
        "sliders": [dict(s, value=vals[s["id"]]) for s in SLIDERS],
        "presets": [{k: p[k] for k in ("id", "label", "blurb")} for p in PRESETS],
        "preset": st.get("preset") or "standard",
        "changed": changed,
        "history": list(st.get("history") or [])[:12],
        "environment": environment_summary(session),
        "rules": gov_rules,
        "team": str(getattr(session, "user_team_name", "") or getattr(session, "user_team_id", "") or ""),
        "season": getattr(session, "season_calendar_year", None),
        "saves": list_saves(),
        "pending_new_sim": bool(_read_pending()),
    }


def apply_preset(session: Any, preset_id: str) -> Dict[str, Any]:
    p = next((x for x in PRESETS if x["id"] == preset_id), None)
    if p is None:
        raise ValueError("Unknown preset")
    return set_values(session, dict(p["values"]), preset=p["label"], replace=True)


# ---------------------------------------------------------------------------
# Disk saves
# ---------------------------------------------------------------------------

_SAVE_LOCK = threading.Lock()


def save_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "saves")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name or "").strip()).strip("-").lower()
    return (s or "franchise")[:48]


def _meta_for(session: Any, name: str) -> Dict[str, Any]:
    return {
        "name": str(name or "Franchise")[:60],
        "team": str(getattr(session, "user_team_name", "") or getattr(session, "user_team_id", "") or ""),
        "team_id": str(getattr(session, "user_team_id", "") or ""),
        "season": getattr(session, "season_calendar_year", None),
        "date": str(getattr(session, "current_date", "") or getattr(session, "calendar_date", "") or ""),
        "phase": str(getattr(session, "phase", "") or ""),
        "saved_at": time.strftime("%Y-%m-%d %H:%M"),
        "sliders_changed": sum(1 for v in (ensure_state(session).get("values") or {}).values() if v is not None),
    }


def save_to_disk(session: Any, name: str = "") -> Dict[str, Any]:
    name = str(name or "").strip() or f"{getattr(session, 'user_team_id', 'team')} {getattr(session, 'season_calendar_year', '')}".strip()
    slot = _slug(name)
    d = save_dir()
    meta = _meta_for(session, name)
    meta["slot"] = slot
    tmp = os.path.join(d, f".{slot}.tmp")
    with _SAVE_LOCK:
        with open(tmp, "wb") as fh:
            pickle.dump(session, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(d, f"{slot}.pkl"))
        with open(os.path.join(d, f"{slot}.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    meta["bytes"] = os.path.getsize(os.path.join(d, f"{slot}.pkl"))
    return {"ok": True, "save": meta, "saves": list_saves()}


def list_saves() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        d = save_dir()
        for fn in os.listdir(d):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            slot = fn[:-5]
            if not os.path.exists(os.path.join(d, f"{slot}.pkl")):
                continue
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {"name": slot}
            meta["slot"] = slot
            out.append(meta)
    except Exception:
        _log.debug("list saves failed", exc_info=True)
    out.sort(key=lambda m: str(m.get("saved_at") or ""), reverse=True)
    return out


def load_from_disk(slot: str) -> Any:
    slot = _slug(slot)
    path = os.path.join(save_dir(), f"{slot}.pkl")
    if not os.path.exists(path):
        raise ValueError("That save no longer exists.")
    with _SAVE_LOCK:
        with open(path, "rb") as fh:
            session = pickle.load(fh)
    try:
        from services.franchise_store import live_code_revision

        session.code_revision = live_code_revision() or str(getattr(session, "code_revision", "") or "")
    except Exception:
        pass
    try:
        session._draft_rankings_build_lock = threading.Lock()
    except Exception:
        pass
    apply(session)
    return session


# ---------------------------------------------------------------------------
# Carry sliders into a new sim
# ---------------------------------------------------------------------------

def _pending_path() -> str:
    return os.path.join(save_dir(), "_next_sim_sliders.json")


def _read_pending() -> Dict[str, Any]:
    try:
        with open(_pending_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def stage_new_sim(session: Optional[Any], carry: bool) -> Dict[str, Any]:
    """Remember (or forget) the current sliders for the next Begin Franchise."""
    path = _pending_path()
    if carry and session is not None:
        st = ensure_state(session)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"values": dict(st.get("values") or {}), "preset": st.get("preset") or "custom"}, fh)
    else:
        try:
            os.replace(path, path + ".used")
        except Exception:
            pass
    return {"ok": True, "carry": bool(carry)}


def adopt_pending(session: Any) -> None:
    """Called once after a new franchise starts: pick up staged sliders, else defaults."""
    data = _read_pending()
    if data:
        st = ensure_state(session)
        st["values"] = {k: _snap(SLIDER_BY_ID[k], v) for k, v in (data.get("values") or {}).items() if k in SLIDER_BY_ID}
        st["preset"] = str(data.get("preset") or "custom")
        st["history"] = [{"date": "", "season": getattr(session, "season_calendar_year", None), "what": "Carried over from the last sim"}]
        try:
            os.replace(_pending_path(), _pending_path() + ".used")
        except Exception:
            pass
    apply(session)
