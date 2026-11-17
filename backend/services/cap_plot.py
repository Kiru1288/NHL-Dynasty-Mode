"""The cap-dump plot: a dark-comedy way to get a bad contract off your books.

You plan it in six choices (motive cover, accomplice, method, alibi, prep time, covering your
tracks). Every choice trades success against suspicion. Then it either works (he "dies in an
accident", the cap hit is gone, and a police inquiry starts) or it fails (he survives, knows
something is off, and suspicion jumps). Investigations resolve over the next weeks; a fixer can
come back months later with a blackmail demand. If you're caught the GM is arrested and the save
is over: every sim refuses to run and you start a new franchise.

Methods are cartoon set-pieces, not instructions. State: ``session.cap_plot``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional
import logging as _logging_swallow

_log = _logging_swallow.getLogger(__name__)

# stage -> [(id, label, blurb, success_mod, suspicion_mod)]
STAGES: Dict[str, List[tuple]] = {
    "cover": [
        ("praise", "Praise him to the press", "\"He's the heart of this team.\" Nobody suspects a fan.", 0.0, -0.10),
        ("silent", "Say nothing", "Keep your head down.", 0.0, 0.0),
        ("rumours", "Float trade rumours first", "Everyone now knows you wanted him gone.", 0.0, 0.14),
    ],
    "accomplice": [
        ("solo", "Do it yourself", "No witnesses. Also no expertise.", -0.12, -0.04),
        ("fixer", "Hire a 'fixer' from a burner number", "Professional. Expensive. Remembers everything.", 0.15, 0.06),
        ("equipment", "Recruit the disgruntled equipment manager", "Cheap, eager, talks to everyone at the bar.", 0.04, 0.12),
    ],
    "method": [
        ("zamboni", "A tragic Zamboni 'malfunction'", "Ice-resurfacing accidents happen. Rarely. Very rarely.", 0.05, 0.08),
        ("bowling", "Team bowling night 'accident'", "Plenty of witnesses, all of them distracted.", -0.05, -0.02),
        ("charter", "Book him on the sketchiest charter flight in the league", "Pilot has 'mostly' passed his checks.", 0.10, 0.05),
        ("motorcycle", "Gift him the world's most dangerous motorcycle", "With a bow on it. And a card signed by you.", 0.02, 0.10),
        ("mascot", "Let the mascot handle it", "Nobody looks under the costume.", -0.15, -0.08),
    ],
    "alibi": [
        ("tv", "Live on TV at the All-Star / board meeting", "Rock-solid. Millions of witnesses.", 0.0, -0.12),
        ("road", "On the road with the team", "Thirty people saw you at the hotel. Probably.", 0.0, -0.05),
        ("home", "Home alone watching film", "Just you and your thoughts. And no alibi.", 0.0, 0.10),
    ],
    "prep": [
        ("rush", "Tonight", "No time to think, no time to slip up. Or to plan.", -0.08, 0.08),
        ("week", "A week of prep", "Measure twice.", 0.04, -0.03),
        ("month", "A month of prep", "Meticulous, but every day is a day someone could talk.", 0.08, -0.06),
    ],
    "tracks": [
        ("burner", "Burner phones, cash only, wipe everything", "Careful people get caught less.", 0.0, -0.10),
        ("bribe", "Pay off the arena security guy", "He seems trustworthy. He said so.", 0.0, -0.04),
        ("none", "Don't bother", "What could go wrong?", 0.0, 0.15),
    ],
}
STAGE_TITLE = {"cover": "Motive cover", "accomplice": "Accomplice", "method": "The 'accident'", "alibi": "Alibi", "prep": "Prep time", "tracks": "Cover your tracks"}
DETECTIVES = ["Det. Sgt. Margot Lavoie", "Inspector Ray Kowalchuk", "Det. Dana Whitlock", "Det. Lt. Marcus Okafor", "Inspector Elise Brunet"]


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def _state(session: Any) -> Dict[str, Any]:
    st = getattr(session, "cap_plot", None)
    if not isinstance(st, dict):
        st = {}
        session.cap_plot = st
    st.setdefault("history", [])
    return st


def _ce():
    from services import contract_economy as ce

    return ce


def _user_team(session: Any) -> Any:
    tid = str(getattr(session, "user_team_id", "") or "")
    tm = (getattr(session, "team_by_id", None) or {}).get(tid)
    if tm is not None:
        return tm
    league = getattr(getattr(session, "sim", None), "league", None)
    for t in list(getattr(league, "teams", None) or []):
        if str(getattr(t, "team_id", "")) == tid:
            return t
    return None


def _find(session: Any, pid: str):
    team = _user_team(session)
    for attr in ("roster", "ahl_roster", "injured_reserve"):
        for p in list(getattr(team, attr, None) or []):
            if str(_ce()._player_id(p)) == str(pid):
                return p, team
    return None, team


def _cap_hit(p: Any) -> float:
    c = getattr(p, "contract", None)
    for k in ("cap_hit_m", "aav_m", "cap_hit", "aav"):
        v = c.get(k) if isinstance(c, dict) else getattr(c, k, None)
        try:
            if v is not None and float(v) > 0:
                f = float(v)
                return f / 1e6 if f > 1000 else f
        except (TypeError, ValueError):
            continue
    return 0.0


def _years_left(p: Any) -> int:
    try:
        return int(_ce()._contract_years_remaining(p) or 0)
    except Exception:
        return 0


def is_game_over(session: Any) -> bool:
    st = getattr(session, "cap_plot", None)
    return bool(isinstance(st, dict) and st.get("game_over"))


def game_over_payload(session: Any) -> Optional[Dict[str, Any]]:
    st = getattr(session, "cap_plot", None)
    if not (isinstance(st, dict) and st.get("game_over")):
        return None
    return dict(st.get("game_over"))


def _motive(session: Any, p: Any) -> float:
    """How obvious a motive you have: big money, long term, under-performing."""
    hit, yrs = _cap_hit(p), _years_left(p)
    try:
        ovr = float(_ce()._player_ovr(p))
    except Exception:
        ovr = 75.0
    value_gap = max(0.0, hit - max(0.8, (ovr - 62) * 0.38))  # $M he's overpaid
    return max(0.0, min(0.30, value_gap * 0.025 + yrs * 0.012))


def options(session: Any, player_id: str) -> Dict[str, Any]:
    if is_game_over(session):
        return {"ok": False, "game_over": game_over_payload(session)}
    p, _team = _find(session, player_id)
    if p is None:
        return {"ok": False, "reason": "He has to be on your roster."}
    st = _state(session)
    blocked = ""
    if st.get("active"):
        blocked = "There's already an open police inquiry. Lie low."
    elif _cap_hit(p) < 0.9:
        blocked = "His cap hit is too small to be worth the risk."
    elif _years_left(p) < 1:
        blocked = "His deal is about to expire anyway."
    last = int(st.get("last_season") or -99)
    if not blocked and int(getattr(session, "season_calendar_year", 0) or 0) - last < 3:
        blocked = "Too soon after the last 'accident'. People would notice a pattern."
    return {
        "ok": True, "player_id": str(player_id), "player_name": str(_ce()._player_name(p)), "cap_hit_m": round(_cap_hit(p), 2),
        "years_left": _years_left(p), "motive": round(_motive(session, p), 3), "blocked": blocked,
        "base_success": 0.62, "base_suspicion": 0.24, "min_suspicion": 0.14,
        "stages": [{"id": k, "title": STAGE_TITLE[k], "options": [{"id": o[0], "label": o[1], "blurb": o[2], "success": o[3], "suspicion": o[4]} for o in v]}
                   for k, v in STAGES.items()],
        "investigation": _investigation_view(session),
    }


def _score(session: Any, p: Any, plan: Dict[str, str]) -> Dict[str, float]:
    succ, susp = 0.62, 0.24 + _motive(session, p)
    for stage, opts in STAGES.items():
        o = next((x for x in opts if x[0] == plan.get(stage)), None)
        if o is None:
            raise ValueError(f"Pick an option for {STAGE_TITLE[stage]}")
        succ += o[3]
        susp += o[4]
    # Combos: a fixer with a burner is cleaner; an equipment manager with no cover-up is a disaster.
    if plan.get("accomplice") == "fixer" and plan.get("tracks") == "burner":
        susp -= 0.04
    if plan.get("accomplice") == "equipment" and plan.get("tracks") == "none":
        susp += 0.10
    if plan.get("cover") == "rumours" and plan.get("alibi") == "home":
        susp += 0.08
    # Even a perfect plan leaves a trail: never better than a 1-in-7 chance of being caught.
    try:
        from app.sim_engine.tuning.command_knobs import knob

        susp *= max(0.3, float(knob("plot_heat", 1.0)))
    except Exception:
        pass
    return {"success": max(0.15, min(0.92, succ)), "suspicion": max(0.14, min(0.95, susp))}


def execute(session: Any, player_id: str, plan: Dict[str, str], confirm: bool = False) -> Dict[str, Any]:
    if is_game_over(session):
        return {"ok": False, "game_over": game_over_payload(session)}
    opts = options(session, player_id)
    if not opts.get("ok"):
        return opts
    if opts.get("blocked"):
        return {"ok": False, "reason": opts["blocked"]}
    if not confirm:
        return {"ok": False, "reason": "Confirm you really want to go through with this."}
    p, team = _find(session, player_id)
    sc = _score(session, p, plan)
    st = _state(session)
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    r = _rng("plot", getattr(session, "session_id", ""), player_id, cur)
    name = str(_ce()._player_name(p))
    st["last_season"] = season
    entry = {"player_id": str(player_id), "name": name, "plan": dict(plan), "success_chance": round(sc["success"], 3),
             "suspicion": round(sc["suspicion"], 3), "season": season, "cursor": cur}
    if r.random() > sc["success"]:
        # He survives. And he's not stupid.
        susp = min(0.95, sc["suspicion"] + 0.25)
        entry.update({"outcome": "survived", "suspicion": round(susp, 3)})
        st["history"].append(entry)
        st["active"] = {"player_id": str(player_id), "name": name, "suspicion": susp, "start": cur, "resolve": cur + r.randint(6, 18),
                        "detective": r.choice(DETECTIVES), "kind": "attempt", "plan": dict(plan)}
        try:
            from services.trade_demand_engine import open_trade_demand

            open_trade_demand(session, p, team, reason="management", calendar_idx=cur, force_formal=True)
        except Exception:
            _log.debug("plot trade demand failed", exc_info=True)
        _story(session, f"Bizarre near-miss for {name}", f"{name} walked away from what police are calling a 'highly unusual' incident. He's asked for a trade.", heat=80)
        session._sim_hard_stop = {"reason": "cap_plot"}
        return {"ok": True, "outcome": "survived", "message": f"It didn't work. {name} survived, he's spooked, and police are asking questions.",
                "investigation": _investigation_view(session)}
    cause = {"zamboni": "an off-ice training accident", "bowling": "an off-ice training accident", "charter": "a private plane crash",
             "motorcycle": "a motorcycle accident", "mascot": "a sudden illness"}.get(plan.get("method"), "a sudden illness")
    _kill(session, p, team, cause)
    entry["outcome"] = "done"
    st["history"].append(entry)
    st["active"] = {"player_id": str(player_id), "name": name, "suspicion": sc["suspicion"], "start": cur, "resolve": cur + r.randint(8, 26),
                    "detective": r.choice(DETECTIVES), "kind": "death", "plan": dict(plan)}
    if plan.get("accomplice") == "fixer" or plan.get("accomplice") == "equipment":
        st["blackmail_due"] = cur + r.randint(50, 120)
        st["blackmailer"] = "the fixer" if plan.get("accomplice") == "fixer" else "the equipment manager"
    return {"ok": True, "outcome": "done", "message": f"It's done. {name}'s cap hit is off your books. Now the police have questions.",
            "cap_freed_m": round(_cap_hit(p), 2), "investigation": _investigation_view(session)}


def _kill(session: Any, p: Any, team: Any, cause: str) -> None:
    pid = str(_ce()._player_id(p))
    fields: Dict[str, Any] = {}
    hit = _cap_hit(p)
    try:
        from services.player_mourning import on_player_death, popup_fields

        fields = popup_fields(on_player_death(session, p, team, cause, suspicious=True, source="plot"))
    except Exception:
        _log.debug("mourning failed", exc_info=True)
    try:
        from app.sim_engine.trades.trade_executor import _purge_player_id_from_team_lists

        _purge_player_id_from_team_lists(team, pid)
    except Exception:
        for attr in ("roster", "ahl_roster", "injured_reserve"):
            setattr(team, attr, [x for x in list(getattr(team, attr, None) or []) if x is not p])
    try:
        p.retired = True
        p.deceased = True
        p.death_cause = cause
        p.active_roster = False
        p.contract = {"years_remaining": 0, "aav_m": 0.0, "cap_hit_m": 0.0, "terminated": "deceased"}
    except Exception:
        pass
    try:
        from services.contract_economy import sync_team_cap_fields

        sync_team_cap_fields(team, getattr(getattr(session, "sim", None), "league", None))
    except Exception:
        _log.debug("cap sync failed", exc_info=True)
    try:
        from services.franchise_sim import _append_showcase_popup, invalidate_session_payload_caches

        name = str(_ce()._player_name(p))
        _append_showcase_popup(session, f"player_death:{pid}:plot", {
            "kind": "player_death", "theme": "obituary", "source_label": "League announcement",
            "headline": f"{name} has died", "summary": fields.get("cause_story") or f"{name} has died.",
            "player_name": name, "player_id": pid, "cause": cause, "team_name": str(getattr(team, "name", "") or ""),
            "is_user_team": True, "interrupt_sim": True, "cap_freed_m": round(hit, 2), **fields,
        })
        invalidate_session_payload_caches(session, "cap_plot")
    except Exception:
        _log.debug("popup failed", exc_info=True)


def _story(session: Any, headline: str, summary: str, heat: int = 70) -> None:
    try:
        from app.sim_engine.franchise.state import _record_storyline

        _record_storyline(session, {"headline": headline, "summary": summary, "team_id": str(getattr(session, "user_team_id", "") or ""),
                                    "category": "major_news", "type": "investigation", "priority": "HIGH", "heat": heat})
    except Exception:
        _log.debug("story failed", exc_info=True)


def _investigation_view(session: Any) -> Optional[Dict[str, Any]]:
    st = getattr(session, "cap_plot", None) or {}
    a = st.get("active") if isinstance(st, dict) else None
    if not a:
        return None
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    total = max(1, int(a["resolve"]) - int(a["start"]))
    return {"player": a["name"], "detective": a["detective"], "heat_pct": int(round(float(a["suspicion"]) * 100)),
            "progress_pct": int(max(0, min(100, (cur - int(a["start"])) / total * 100))), "kind": a["kind"]}


def _arrest(session: Any, why: str) -> None:
    st = _state(session)
    a = st.get("active") or {}
    st["game_over"] = {
        "headline": "GM arrested",
        "summary": f"{a.get('detective', 'Detectives')} arrested you this morning. {why} The league has stripped the franchise from you. This save is over.",
        "player": a.get("name"), "season": int(getattr(session, "season_calendar_year", 0) or 0),
    }
    st["active"] = None
    session._sim_hard_stop = {"reason": "game_over"}
    _story(session, "GM arrested in connection with player's death", st["game_over"]["summary"], heat=100)
    try:
        from services.franchise_sim import _append_showcase_popup, invalidate_session_payload_caches

        _append_showcase_popup(session, "cap_plot_game_over", {"kind": "breaking_news", "headline": "GM ARRESTED", "summary": st["game_over"]["summary"],
                                                              "source_label": "Breaking news", "interrupt_sim": True, "game_over": True})
        invalidate_session_payload_caches(session, "cap_plot")
    except Exception:
        pass


def daily_tick(session: Any) -> None:
    st = getattr(session, "cap_plot", None)
    if not isinstance(st, dict) or st.get("game_over"):
        return
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    a = st.get("active")
    if a and cur >= int(a["resolve"]):
        r = _rng("verdict", getattr(session, "session_id", ""), a["player_id"], a["resolve"])
        if r.random() < float(a["suspicion"]):
            plan = a.get("plan") or {}
            why = {"equipment": "The equipment manager confessed at the bar.", "fixer": "The fixer flipped for a plea deal.",
                   "solo": "Your fingerprints were all over it."}.get(plan.get("accomplice"), "The evidence pointed to you.")
            if plan.get("tracks") == "none":
                why = "Your phone records did most of the work."
            _arrest(session, why)
            return
        st["active"] = None
        _story(session, f"Police close the inquiry into {a['name']}", "Investigators found nothing they could prove. The case is closed. For now.", heat=60)
    due = st.get("blackmail_due")
    if due and cur >= int(due) and not st.get("blackmail_open"):
        st["blackmail_open"] = True
        pend = getattr(session, "pending_decisions", None)
        if not isinstance(pend, list):
            session.pending_decisions = pend = []
        who = st.get("blackmailer") or "someone"
        pend.append({"id": f"cap_plot_blackmail_{cur}", "kind": "cap_plot_blackmail", "requires_action": True, "title": "An envelope on your desk",
                     "description": f"A note from {who}: 'I remember everything. Pay up or the police get the whole story.'",
                     "options": [{"id": "pay", "label": "Pay quietly", "effect_summary": "Probably the end of it. Probably."},
                                 {"id": "refuse", "label": "Call the bluff", "effect_summary": "Maybe they won't talk."},
                                 {"id": "silence", "label": "Arrange a 'retirement' for them too", "effect_summary": "Two accidents in one year is a pattern."}]})
        session._sim_hard_stop = {"reason": "cap_plot"}


def resolve_plot_decision(session: Any, d: Dict[str, Any], cid: str) -> Dict[str, Any]:
    st = _state(session)
    st["blackmail_due"] = None
    st["blackmail_open"] = False
    r = _rng("blackmail", getattr(session, "session_id", ""), d.get("id"), cid)
    caught = {"pay": 0.05, "refuse": 0.35, "silence": 0.55}.get(cid, 0.2)
    if r.random() < caught:
        st["active"] = {"player_id": "", "name": "the blackmail case", "suspicion": 1.0, "start": 0, "resolve": 0,
                        "detective": r.choice(DETECTIVES), "kind": "blackmail", "plan": {}}
        _arrest(session, {"pay": "The payment was traced.", "refuse": "They went to the police.", "silence": "The second 'accident' gave you away."}[cid])
        return {"arrested": True}
    _story(session, "Quiet week in the front office", "Nothing to see here.", heat=40)
    return {"arrested": False}
