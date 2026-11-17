"""Team disaster and the NHL Emergency Rehabilitation Plan ("disaster draft").

Real rule (NHL by-laws, Emergency Rehabilitation Plan): it activates when five or more players of
one club are killed or disabled. The club first buys contracts from willing clubs (paid from a
league insurance fund). If it still lacks a playing roster, an Emergency Rehabilitation Draft is
held: every other club protects one goalie and ten skaters, no club loses more than one player,
and clubs that sold a contract are exempt.

In the franchise the draft refills every lost spot (not just the bare minimum) so the club can
actually compete. Extremely rare: about once in 90 seasons.

State: ``session.disaster_draft`` while picks are owed by the user, ``session.disasters`` history.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple
import logging as _logging_swallow

_log = _logging_swallow.getLogger(__name__)

DAILY_CHANCE = 0.00006


def _knob(name: str) -> float:
    try:
        from app.sim_engine.tuning.command_knobs import knob

        return max(0.0, float(knob(name, 1.0)))
    except Exception:
        return 1.0
INCIDENTS = [
    ("the team charter went down on approach in a snowstorm", "a private plane crash"),
    ("the team bus left the highway in an ice storm on the way to the airport", "a car accident"),
    ("a fire swept through the team hotel overnight", "a house fire"),
]


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def _ce():
    from services import contract_economy as ce

    return ce


def _pid(p: Any) -> str:
    return str(_ce()._player_id(p))


def _name(p: Any) -> str:
    return str(_ce()._player_name(p))


def _ovr(p: Any) -> float:
    try:
        return float(_ce()._player_ovr(p))
    except Exception:
        return 60.0


def _is_g(p: Any) -> bool:
    try:
        return str(_ce()._position_bucket(p)) == "G"
    except Exception:
        return False


def _tid(t: Any) -> str:
    return str(getattr(t, "team_id", None) or getattr(t, "id", "") or "")


def _tname(t: Any) -> str:
    return str(getattr(t, "name", None) or _tid(t))


def _teams(session: Any) -> List[Any]:
    return list(getattr(getattr(getattr(session, "sim", None), "league", None), "teams", None) or [])


def _team(session: Any, tid: str) -> Any:
    return next((t for t in _teams(session) if _tid(t) == str(tid)), None)


def _story(session: Any, headline: str, summary: str, team_id: str = "", heat: int = 90) -> None:
    try:
        from app.sim_engine.franchise.state import _record_storyline

        _record_storyline(session, {"headline": headline, "summary": summary, "team_id": team_id, "category": "major_news",
                                    "type": "disaster", "priority": "HIGH", "heat": heat})
    except Exception:
        _log.debug("story failed", exc_info=True)
    try:
        session.timeline.append(headline)
    except Exception:
        pass


def _has_nmc(p: Any) -> bool:
    c = getattr(p, "contract", None)
    for k in ("nmc", "no_movement", "no_movement_clause"):
        v = c.get(k) if isinstance(c, dict) else getattr(c, k, None)
        if v:
            return True
    return False


def _sync(session: Any, *teams: Any) -> None:
    try:
        from services.contract_economy import sync_team_cap_fields

        league = getattr(getattr(session, "sim", None), "league", None)
        for t in teams:
            if t is not None:
                sync_team_cap_fields(t, league)
    except Exception:
        _log.debug("cap sync failed", exc_info=True)


def _move(session: Any, p: Any, src: Any, dst: Any) -> None:
    src.roster = [x for x in list(getattr(src, "roster", None) or []) if x is not p]
    dst.roster = list(getattr(dst, "roster", None) or []) + [p]
    for attr in ("team_id", "current_team_id"):
        try:
            setattr(p, attr, _tid(dst))
        except Exception:
            pass
    _sync(session, src, dst)


def maybe_disaster(session: Any) -> None:
    phase = str(getattr(session, "phase", "") or "").lower()
    if phase not in ("regular", "preseason"):
        return
    if getattr(session, "disaster_draft", None):
        return
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    if _rng("disaster", getattr(session, "session_id", ""), cur).random() >= DAILY_CHANCE * _knob("disaster_rate"):
        return
    trigger_disaster(session)


def trigger_disaster(session: Any, team_id: Optional[str] = None) -> Dict[str, Any]:
    cur = int(getattr(session, "calendar_cursor", 0) or 0)
    r = _rng("disaster_pick", getattr(session, "session_id", ""), cur, team_id or "")
    teams = _teams(session)
    team = _team(session, team_id) if team_id else (r.choice(teams) if teams else None)
    if team is None:
        return {"ok": False}
    roster = list(getattr(team, "roster", None) or [])
    if len(roster) < 10:
        return {"ok": False}
    incident, cause = r.choice(INCIDENTS)
    n = r.randint(5, 8)
    victims = r.sample(roster, min(n, len(roster) - 4))
    killed = victims[: max(3, int(round(len(victims) * 0.7)))]
    disabled = [v for v in victims if v not in killed]
    names_k = [_name(p) for p in killed]
    names_d = [_name(p) for p in disabled]
    lost_goalie = any(_is_g(p) for p in victims)

    # Mourning: the first victim carries the club's decisions; the others are remembered with him.
    try:
        from services.player_mourning import on_player_death

        for i, p in enumerate(killed):
            on_player_death(session, p, team, cause, quiet=i > 0)
    except Exception:
        _log.debug("mourning failed", exc_info=True)
    for p in killed + disabled:
        try:
            p.retired = True
            p.career_ended = True
            p.active_roster = False
            p.contract = {"years_remaining": 0, "aav_m": 0.0, "cap_hit_m": 0.0, "terminated": "deceased" if p in killed else "disabled"}
            if p in killed:
                p.deceased = True
                p.death_cause = cause
        except Exception:
            pass
    team.roster = [x for x in roster if x not in victims]
    _sync(session, team)
    tname = _tname(team)
    _story(session, f"Tragedy: {tname} lose {len(killed)} players", f"Last night {incident}. Killed: {', '.join(names_k)}."
           + (f" Career-ending injuries: {', '.join(names_d)}." if names_d else "") + " The hockey world is in mourning.", team_id=_tid(team), heat=100)
    _story(session, "League activates the Emergency Rehabilitation Plan",
           f"With {len(victims)} players lost, the {tname} will first buy contracts from willing clubs, then hold an emergency draft. "
           "Every other club protects one goalie and ten skaters and can lose at most one player.", heat=92)
    session._sim_hard_stop = {"reason": "disaster"}

    # Phase 1: contract purchases from willing clubs (insurance fund pays).
    others = [t for t in teams if t is not team]
    r.shuffle(others)
    exempt = set()
    bought = []
    for seller in others[: r.randint(1, 2)]:
        depth = sorted((p for p in list(getattr(seller, "roster", None) or []) if not _has_nmc(p)), key=_ovr, reverse=True)[13:19]
        if not depth:
            continue
        p = r.choice(depth)
        _move(session, p, seller, team)
        exempt.add(_tid(seller))
        bought.append({"name": _name(p), "from": _tname(seller)})
    if bought:
        _story(session, f"{tname} buy contracts from willing clubs", "; ".join(f"{b['name']} from the {b['from']}" for b in bought) + ". The league insurance fund pays.", team_id=_tid(team), heat=70)

    picks = max(0, len(victims) - len(bought))
    rec = {"team_id": _tid(team), "team_name": tname, "cursor": cur, "season": int(getattr(session, "season_calendar_year", 0) or 0),
           "incident": incident, "killed": names_k, "disabled": names_d, "bought": bought, "exempt": sorted(exempt),
           "picks_total": picks, "picks_left": picks, "taken_from": [], "picks": [], "need_goalie": lost_goalie}
    hist = list(getattr(session, "disasters", None) or [])
    hist.append(rec)
    session.disasters = hist[-10:]
    user = str(getattr(session, "user_team_id", "") or "") == _tid(team)
    if picks and user:
        session.disaster_draft = rec  # the user makes the picks (screen blocks until done)
    elif picks:
        _auto_draft(session, rec)
    return {"ok": True, "team": tname, "killed": names_k, "disabled": names_d, "picks": picks}


def protected_ids(team: Any) -> set:
    roster = list(getattr(team, "roster", None) or [])
    gs = sorted((p for p in roster if _is_g(p)), key=_ovr, reverse=True)[:1]
    sk = sorted((p for p in roster if not _is_g(p)), key=_ovr, reverse=True)[:10]
    return {_pid(p) for p in gs + sk} | {_pid(p) for p in roster if _has_nmc(p)}


def available_pool(session: Any, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    blocked = set(rec.get("exempt") or []) | set(rec.get("taken_from") or []) | {rec["team_id"]}
    for t in _teams(session):
        if _tid(t) in blocked:
            continue
        prot = protected_ids(t)
        for p in list(getattr(t, "roster", None) or []):
            if _pid(p) in prot:
                continue
            c = getattr(p, "contract", None)
            hit = (c.get("cap_hit_m") or c.get("aav_m")) if isinstance(c, dict) else getattr(c, "cap_hit_m", None)
            try:
                age = int(_ce()._player_age(p) or 0)
            except Exception:
                age = 0
            out.append({"player_id": _pid(p), "name": _name(p), "team_id": _tid(t), "team": _tname(t),
                        "position": str(_ce()._position_bucket(p) or ""), "ovr": round(_ovr(p), 1), "age": age,
                        "cap_hit_m": round(float(hit or 0), 2)})
    out.sort(key=lambda x: -x["ovr"])
    return out


def _take(session: Any, rec: Dict[str, Any], player_id: str) -> Dict[str, Any]:
    pool = {x["player_id"]: x for x in available_pool(session, rec)}
    row = pool.get(str(player_id))
    if row is None:
        return {"ok": False, "reason": "That player isn't available (protected, or his club already lost one)."}
    src = _team(session, row["team_id"])
    dst = _team(session, rec["team_id"])
    p = next((x for x in list(getattr(src, "roster", None) or []) if _pid(x) == str(player_id)), None)
    if p is None or dst is None:
        return {"ok": False, "reason": "Player not found."}
    _move(session, p, src, dst)
    rec["taken_from"].append(row["team_id"])
    rec["picks"].append(row)
    rec["picks_left"] = max(0, int(rec["picks_left"]) - 1)
    if _is_g(p):
        rec["need_goalie"] = False
    if str(getattr(session, "user_team_id", "") or "") == row["team_id"]:
        _story(session, f"You lose {row['name']} in the emergency draft", f"The {rec['team_name']} selected him from your unprotected list.",
               team_id=row["team_id"], heat=70)
    return {"ok": True, "pick": row, "picks_left": rec["picks_left"]}


def _auto_draft(session: Any, rec: Dict[str, Any]) -> None:
    while int(rec["picks_left"]) > 0:
        pool = available_pool(session, rec)
        if not pool:
            break
        if rec.get("need_goalie"):
            g = [x for x in pool if x["position"] == "G"]
            pick = g[0] if g else pool[0]
        else:
            sk = [x for x in pool if x["position"] != "G"]
            pick = sk[0] if sk else pool[0]
        if not _take(session, rec, pick["player_id"]).get("ok"):
            break
    _finish(session, rec)


def _finish(session: Any, rec: Dict[str, Any]) -> None:
    if rec.get("picks"):
        _story(session, f"{rec['team_name']} complete the emergency draft",
               "Selected: " + "; ".join(f"{x['name']} ({x['team']})" for x in rec["picks"]) + ".", team_id=rec["team_id"], heat=80)
    if getattr(session, "disaster_draft", None) is rec:
        session.disaster_draft = None


def draft_payload(session: Any) -> Dict[str, Any]:
    rec = getattr(session, "disaster_draft", None)
    if not rec:
        return {"ok": True, "active": False, "history": list(getattr(session, "disasters", None) or [])[-3:]}
    return {"ok": True, "active": True, **{k: rec[k] for k in ("team_name", "incident", "killed", "disabled", "bought", "picks_total", "picks_left", "picks", "need_goalie")},
            "pool": available_pool(session, rec)[:120],
            "rules": "Every other club protected one goalie and ten skaters. You can take at most one player from each club; clubs that sold you a contract are exempt."}


def make_pick(session: Any, player_id: str) -> Dict[str, Any]:
    rec = getattr(session, "disaster_draft", None)
    if not rec:
        return {"ok": False, "reason": "No emergency draft is running."}
    res = _take(session, rec, player_id)
    if res.get("ok") and int(rec["picks_left"]) <= 0:
        _finish(session, rec)
    try:
        from services.franchise_sim import invalidate_session_payload_caches

        invalidate_session_payload_caches(session, "disaster_draft")
    except Exception:
        pass
    return {**res, "draft": draft_payload(session)}


def auto_finish(session: Any) -> Dict[str, Any]:
    rec = getattr(session, "disaster_draft", None)
    if rec:
        _auto_draft(session, rec)
    return draft_payload(session)
