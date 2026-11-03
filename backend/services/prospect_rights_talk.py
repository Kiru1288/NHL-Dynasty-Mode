"""Talking to rights-held prospects (junior / college / Europe) and their signing interest.

* ``signing_interest(session, player, team)``: 0-100 read on how keen he is to sign with
  the club that holds his rights. Built from character, how good the NHL club is, the depth
  chart at his position, his relationship with the org (``org_relationship`` plus rapport
  built in talks: ``_rights_talk_rapport``), his agent's style and his own willingness.
* NCAA refusal (rare, real precedent): a college player can decline to sign and become a
  free agent if unsigned by Aug 15 after his class graduates. Once per season an NCAA
  prospect whose interest is very low may form ``_ncaa_refusal_intent``. While it holds,
  ``elc_offer_engine.evaluate_offer_acceptance`` caps the user's offers well below the
  acceptance line, and ``rights_can_reenter_draft`` is set False so the existing Aug 15
  expiry in ``draft_rights_engine.process_draft_rights_deadlines`` makes him a UFA.
  Good conversations rebuild rapport and can talk him out of it.
* ``talk(session, player_id, topic, approach)``: check-in conversation. Topics: how he's
  feeling, development, league & role, signing intentions, turn pro vs stay in school/junior.
  His answers read his stats, development and personality; the GM's approach moves rapport.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

import logging as _logging_swallow

_swallowed_log = _logging_swallow.getLogger(__name__)

TALK_COOLDOWN_DAYS = 14
REFUSAL_INTEREST_LINE = 34.0  # below this an NCAA kid may form refusal intent (rare)
REFUSAL_CLEAR_LINE = 50.0  # rapport rebuilt past this -> intent dropped
WARNING_LINE = 45.0

TOPICS = (
    ("feeling", "How are you feeling?"),
    ("development", "Talk about your development"),
    ("role", "Your league and role"),
    ("signing", "Where's your head at on signing?"),
    ("pro_or_stay", "Turn pro or stay another year?"),
)


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:12], 16))


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _i(v: Any) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def _day(session: Any) -> int:
    return _i(getattr(session, "calendar_cursor", 0))


def _season(session: Any) -> int:
    return _i(getattr(session, "season_calendar_year", 0))


def _pid(player: Any) -> str:
    return str(getattr(player, "id", None) or getattr(player, "player_id", "") or "")


def _name(player: Any) -> str:
    n = getattr(player, "name", None)
    if not n:
        ident = getattr(player, "identity", None)
        n = getattr(ident, "name", None) or " ".join(
            str(x) for x in (getattr(ident, "first_name", ""), getattr(ident, "last_name", "")) if x
        )
    return str(n or "Prospect")


def _personality(player: Any) -> Dict[str, float]:
    try:
        from app.sim_engine.franchise.storyline_engine import _u_personality  # noqa: WPS433

        return dict(_u_personality(player, _pid(player)))
    except Exception:
        return {}


def _path_bucket(player: Any) -> str:
    try:
        from services.draft_rights_engine import _prospect_path_bucket

        return _prospect_path_bucket(player)
    except Exception:
        return "other"


def _pos_bucket(player: Any) -> str:
    try:
        from services.draft_rights_engine import _signing_position_bucket

        return _signing_position_bucket(player)
    except Exception:
        return "F"


def _ovr(player: Any) -> float:
    try:
        from services.draft_rights_engine import _safe_player_ovr

        return float(_safe_player_ovr(player))
    except Exception:
        return 0.0


def _age(player: Any) -> int:
    try:
        from services.draft_rights_engine import _safe_player_age

        return int(_safe_player_age(player))
    except Exception:
        return 19


def _user_team(session: Any) -> Any:
    return (getattr(session, "team_by_id", None) or {}).get(str(getattr(session, "user_team_id", "") or ""))


def _is_user_team(session: Any, team: Any) -> bool:
    if session is None or team is None:
        return False
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    return bool(tid) and tid == str(getattr(session, "user_team_id", "") or "")


def _season_line(player: Any) -> Dict[str, Any]:
    st = getattr(player, "_prospect_season_stats", None)
    st = st if isinstance(st, dict) else {}
    gp = _i(st.get("gp") or st.get("games_played"))
    g, a = _i(st.get("goals")), _i(st.get("assists"))
    pts = _i(st.get("points")) or g + a
    return {"gp": gp, "g": g, "a": a, "pts": pts, "ppg": round(pts / gp, 2) if gp else 0.0,
            "sv_pct": st.get("sv_pct") or st.get("save_pct"), "gaa": st.get("gaa"),
            "league": str(st.get("league") or getattr(player, "current_league_id", "") or getattr(player, "development_path", "") or ""),
            "team": str(st.get("team") or "")}


def _ovr_delta(player: Any) -> float:
    start = getattr(player, "_season_start_ovr", None)
    if start is None:
        start = getattr(player, "season_start_ovr", None)
    if start is None:
        return 0.0
    return round(_ovr(player) - _f(start), 1)


def _team_strength(session: Any, team: Any) -> Tuple[float, str]:
    """0..1 (1 = best club) and a short label."""
    tid = str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")
    recs = getattr(getattr(session, "standings", None), "records", None) or {}
    rows: List[Tuple[str, float, int]] = []
    if isinstance(recs, dict):
        for k, r in recs.items():
            w, l, o = _i(getattr(r, "wins", 0)), _i(getattr(r, "losses", 0)), _i(getattr(r, "otl", 0))
            gp = w + l + o
            pts = _i(getattr(r, "points", 2 * w + o))
            rows.append((str(k), pts / max(1, 2 * gp), gp))
    if rows and next((gp for k, _, gp in rows if k == tid), 0) >= 10:
        rows.sort(key=lambda r: -r[1])
        rank = next((i for i, r in enumerate(rows) if r[0] == tid), len(rows) // 2)
    else:
        hist = [h for h in list(getattr(session, "season_history", None) or []) if isinstance(h, dict)]
        last = (hist[-1].get("teams") or {}) if hist else {}
        row = last.get(tid) if isinstance(last, dict) else None
        if not isinstance(row, dict) or not row.get("league_rank"):
            return 0.5, "middle of the pack"
        rank = _i(row.get("league_rank")) - 1
        rows = [None] * max(len(last), 1)  # type: ignore
    n = max(1, len(rows) - 1)
    strength = 1.0 - rank / n
    label = "a contender" if strength >= 0.72 else "a bubble team" if strength >= 0.42 else "a rebuilding team"
    return round(strength, 2), label


def _depth_crowd(team: Any, player: Any) -> Tuple[int, str]:
    """How many established NHL players at his position are at or above his projection."""
    pos = _pos_bucket(player)
    proj = max(_ovr(player), _f(getattr(player, "potential_display", None) or getattr(player, "potential", None), 0.0))
    proj = proj * 99.0 if 0 < proj <= 1.5 else proj
    target = min(proj, _ovr(player) + 8)
    n = 0
    for p in list(getattr(team, "roster", None) or []):
        if _pos_bucket(p) != pos:
            continue
        if _age(p) <= 31 and _ovr(p) >= target:
            n += 1
    limit = {"F": 9, "D": 5, "G": 2}.get(pos, 9)
    label = "crowded" if n >= limit else "open" if n <= limit - 3 else "competitive"
    return n, label


def _agent(player: Any, session: Any) -> Dict[str, Any]:
    try:
        from app.sim_engine.franchise.player_agent_engine import agent_public_view  # noqa: WPS433

        return dict(agent_public_view(player, session) or {})
    except Exception:
        return {}


def signing_interest(session: Any, player: Any, team: Any = None) -> Dict[str, Any]:
    team = team if team is not None else _user_team(session)
    pers = _personality(player)
    character = _f(pers.get("character"), 55)
    ego = _f(pers.get("ego"), 50)
    ambition = _f(pers.get("ambition"), 55)
    loyalty = _f(pers.get("loyalty"), 55)
    path = _path_bucket(player)
    factors: List[Dict[str, Any]] = []

    def add(label: str, delta: float) -> None:
        if abs(delta) >= 0.5:
            factors.append({"label": label, "delta": round(delta, 1)})

    score = 68.0
    if character < 48:
        add("Questionable character", -(48 - character) * 0.7)
    elif character >= 68:
        add("High character", 3.0)
    strength, team_label = _team_strength(session, team) if team is not None else (0.5, "")
    team_delta = (strength - 0.5) * 18.0
    if ambition >= 62:
        team_delta *= 1.4
    add(f"Club is {team_label}" if team_label else "Club outlook", team_delta)
    crowd, crowd_label = _depth_crowd(team, player) if team is not None else (0, "")
    pos = _pos_bucket(player)
    limit = {"F": 9, "D": 5, "G": 2}.get(pos, 9)
    if crowd_label == "crowded":
        add(f"Depth chart is crowded at {pos}", -6.0 - (crowd - limit) * 2.0 - max(0.0, ambition - 60) * 0.15)
    elif crowd_label == "open":
        add(f"Clear path at {pos}", 5.0)
    rel = _f(getattr(player, "org_relationship", None), 0.55)
    add("Relationship with the org", (rel - 0.55) * 40.0)
    rapport = _f(getattr(player, "_rights_talk_rapport", 0.0), 0.0)
    add("Your conversations with him", rapport)
    agent = _agent(player, session)
    style = str(agent.get("style") or "")
    if style in ("leverage", "disruptor"):
        add(f"Agent ({agent.get('style_label') or style}) plays the market", -6.0 if style == "leverage" else -9.0)
    elif style == "discreet":
        add("Patient agent", 2.0)
    will = getattr(player, "willingness_to_sign", None)
    if will is True:
        add("Wants to turn pro", 8.0)
    elif will is False:
        add("Not ready to commit", -10.0)
    if ego >= 70:
        add("Big ego", -(ego - 70) * 0.25)
    add("Loyal by nature", (loyalty - 55) * 0.12)
    score += sum(f["delta"] for f in factors)
    interest = round(_clip(score), 1)

    refuses = bool(getattr(player, "_ncaa_refusal_intent", False))
    if path == "ncaa" and session is not None:
        refuses = _update_refusal_intent(session, player, interest)
    label = ("Eager to sign" if interest >= 75 else "Open to signing" if interest >= 58
             else "Lukewarm" if interest >= WARNING_LINE else "Leaning against signing")
    if refuses:
        label = "Plans to test free agency"
    warning = None
    if refuses:
        warning = (f"{_name(player)} has told people he won't sign here. If he's unsigned when his rights lapse "
                   f"(Aug 15 after his class graduates), he walks as a free agent. Talk to him to change his mind.")
    elif interest < WARNING_LINE:
        warning = f"{_name(player)} is leaning toward not signing. A conversation could turn it around."
    factors.sort(key=lambda f: f["delta"])
    try:
        player._signing_interest = interest
    except Exception:
        pass
    return {
        "player_id": _pid(player),
        "interest": interest,
        "label": label,
        "tone": "bad" if refuses or interest < WARNING_LINE else "warn" if interest < 58 else "good",
        "leaning_no": bool(refuses or interest < WARNING_LINE),
        "refuses": refuses,
        "warning": warning,
        "path": path,
        "factors": factors[:6],
        "rapport": round(rapport, 1),
    }


def _update_refusal_intent(session: Any, player: Any, interest: float) -> bool:
    intent = bool(getattr(player, "_ncaa_refusal_intent", False))
    if intent and interest >= REFUSAL_CLEAR_LINE:
        _set_refusal(player, False)
        return False
    if intent:
        return True
    season = _season(session)
    if _i(getattr(player, "_ncaa_refusal_rolled_season", 0)) == season:
        return False
    try:
        player._ncaa_refusal_rolled_season = season
    except Exception:
        return False
    if interest >= REFUSAL_INTEREST_LINE or _age(player) < 19:
        return False
    # Very rare: even at rock-bottom interest it's about a one-in-three roll.
    chance = min(0.35, (REFUSAL_INTEREST_LINE - interest) / REFUSAL_INTEREST_LINE * 0.6)
    if _rng(_pid(player), season, "ncaa_refuse").random() < chance:
        _set_refusal(player, True)
        return True
    return False


def _set_refusal(player: Any, on: bool) -> None:
    try:
        if on:
            if getattr(player, "_ncaa_refusal_prev_reenter", None) is None:
                player._ncaa_refusal_prev_reenter = bool(getattr(player, "rights_can_reenter_draft", True))
            player._ncaa_refusal_intent = True
            player.rights_can_reenter_draft = False  # Aug 15 expiry -> UFA, not re-entry
        else:
            player._ncaa_refusal_intent = False
            prev = getattr(player, "_ncaa_refusal_prev_reenter", None)
            if prev is not None:
                player.rights_can_reenter_draft = bool(prev)
            player._ncaa_refusal_prev_reenter = None
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)


def elc_signing_adjustment(player: Any, team: Any) -> Dict[str, Any]:
    """Used by elc_offer_engine.evaluate_offer_acceptance. Only the user's club is affected."""
    session = getattr(team, "_session_ref", None)
    if session is None or not _is_user_team(session, team):
        return {}
    try:
        si = signing_interest(session, player, team)
    except Exception:
        return {}
    out: Dict[str, Any] = {"score_delta": round((si["interest"] - 60.0) / 100.0 * 0.3, 3), "interest": si["interest"]}
    if si["refuses"]:
        out["cap"] = 0.2
        out["concern"] = "intends to become a free agent rather than sign (Aug 15 rule)"
    elif si["interest"] < WARNING_LINE:
        out["concern"] = "leaning toward not signing with you"
    elif si["interest"] >= 75:
        out["positive"] = "eager to sign with your club"
    return out


# ---------------------------------------------------------------------------
# Rights list
# ---------------------------------------------------------------------------

def _rights_players(team: Any) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for e in list(getattr(team, "reserve_list", None) or []):
        p = e.get("player_ref") if isinstance(e, dict) else None
        if p is not None and _pid(p) not in seen:
            seen.add(_pid(p))
            out.append(p)
    for p in list(getattr(team, "prospect_pool", None) or []):
        if _pid(p) and _pid(p) not in seen:
            seen.add(_pid(p))
            out.append(p)
    return [p for p in out if str(getattr(p, "signed_status", "") or "").lower() != "signed"]


def interest_board(session: Any) -> Dict[str, Any]:
    team = _user_team(session)
    rows: Dict[str, Any] = {}
    if team is None:
        return {"players": rows}
    for p in _rights_players(team):
        try:
            rows[_pid(p)] = signing_interest(session, p, team)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    return {"players": rows}


def _find(session: Any, player_id: str) -> Any:
    team = _user_team(session)
    if team is None:
        return None
    try:
        from services.contract_economy import find_prospect_on_team

        p = find_prospect_on_team(team, str(player_id))
        if p is not None:
            return p
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return next((p for p in _rights_players(team) if _pid(p) == str(player_id)), None)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def _pick(seq: Any, *seed: Any) -> str:
    seq = list(seq)
    return seq[_rng(*seed).randrange(len(seq))] if seq else ""


def _league_word(path: str) -> str:
    return {"ncaa": "school", "junior": "junior", "europe": "over there"}.get(path, "where I am")


def _mood(player: Any, si: Dict[str, Any]) -> float:
    line = _season_line(player)
    d = _ovr_delta(player)
    pers = _personality(player)
    m = 55.0 + d * 3.0 + (line["ppg"] - 0.7) * 18.0 + (si["interest"] - 60) * 0.25
    m += (_f(pers.get("resilience"), 55) - 55) * 0.15
    return _clip(m)


def _answer(session: Any, player: Any, topic: str, si: Dict[str, Any]) -> str:
    pid, day = _pid(player), _day(session)
    path = si["path"]
    line = _season_line(player)
    d = _ovr_delta(player)
    age = _age(player)
    ovr = _ovr(player)
    goalie = _pos_bucket(player) == "G"
    stat = (f"{line['sv_pct']:.3f} save percentage" if goalie and line.get("sv_pct") else
            f"{line['pts']} points in {line['gp']} games" if line["gp"] else "not many games yet")
    seed = (pid, day, topic)
    if topic == "feeling":
        m = _mood(player, si)
        if m >= 66:
            return _pick(("Honestly? Best I've felt. Things are clicking.", "Really good. I'm having fun again.", "Great. I wake up wanting to get to the rink."), *seed) + f" {stat.capitalize()} so far."
        if m >= 46:
            return _pick(("Good, mostly. Some up and down.", "Okay. Grinding through it.", "Not bad. Could be better, could be worse."), *seed) + f" I've got {stat}."
        return _pick(("Rough stretch, not gonna lie.", "Tough. Nothing's going in.", "Frustrated, if I'm honest."), *seed) + f" {stat.capitalize()}, and it's eating at me."
    if topic == "development":
        if d >= 2:
            return f"I can feel it. My game's taken a step this year (+{d:g} on the scouting sheet). The extra skating work is paying off."
        if d <= -1:
            return "I've stalled a bit. I know it. I need a better plan, or more from the coaching here."
        return _pick(("Steady. Little things every week.", "It's coming. Not as fast as I want."), *seed) + (" My shot's getting there." if not goalie else " My tracking's better.")
    if topic == "role":
        lg = line["league"] or _league_word(path)
        if line["gp"] and line["ppg"] >= 1.0 and not goalie:
            return f"I'm a top guy in {lg}. Power play, big minutes. Maybe I'm outgrowing it."
        if line["gp"] and line["ppg"] < 0.4 and not goalie:
            return f"Third line, some PK in {lg}. I want a bigger role, but I'm learning to play without the puck."
        return f"Solid minutes in {lg}. Coaches trust me. I'd like more power-play time."
    if topic == "signing":
        if si["refuses"]:
            return _pick(("I'll be honest with you. My agent and I are looking at all our options after school.",
                          "I just don't see where I fit here. I think I'd rather pick my spot."), *seed)
        if si["interest"] >= 75:
            return "Whenever you're ready, I'm ready. I want to be part of this."
        if si["interest"] >= 58:
            return "I'm open to it. I want to see a real path first. Where do I fit?"
        if si["interest"] >= WARNING_LINE:
            return "Not sure yet. I'm keeping my options open."
        worst = (si.get("factors") or [{}])[0].get("label", "")
        return "Honestly, I have doubts" + (f" ({worst.lower()})" if worst else "") + ". I'm not sure this is the right place for me."
    if topic == "pro_or_stay":
        ready = ovr >= 72 or (line["gp"] and line["ppg"] >= 1.1)
        if path == "ncaa":
            if ready and age >= 20:
                return "I think I'm ready to leave school. I've done what I came to do."
            return "I want one more year in school. Get stronger, finish what I started, then turn pro."
        if path == "junior":
            if ready:
                return "I'm ready to turn pro. Junior isn't pushing me anymore."
            return "Another year in junior wouldn't hurt. I want to dominate before I make the jump."
        if path == "europe":
            if ready:
                return "I'm ready to come over. I want to play in the NHL."
            return "My club wants me back, and I'm playing big minutes against men. I think one more year."
        return "I'd turn pro tomorrow if there's a spot." if ready else "I think I need another year."
    return "Okay."


_APPROACHES = {
    "feeling": (("support", "Tell him the org has his back", "Warm. Builds rapport, especially with young or sensitive kids."),
                ("push", "Push him to compete harder", "Tough love. Competitive kids respond; fragile ones don't.")),
    "development": (("plan", "Lay out a development plan with our coaches", "Concrete. Builds trust if he's coachable."),
                    ("praise", "Praise his progress", "Feels good. Rings hollow if he's stalling.")),
    "role": (("role_ok", "Tell him his role is the right one for now", "Honest. Patient players accept it."),
             ("role_more", "Promise to push for a bigger role", "He'll like it. You're on the hook to deliver.")),
    "signing": (("path", "Show him his path to our NHL roster", "Best when the depth chart really is open."),
                ("commit", "Tell him we want him signed now", "Shows you value him. Pressures him."),
                ("respect", "Respect his process, no pressure", "Low risk. Keeps the door open.")),
    "pro_or_stay": (("back_choice", "Back whatever he decides", "Earns trust. He may stay another year."),
                    ("go_pro", "Make the case for turning pro now", "Works if he's ready; pushes him if he isn't.")),
}


def _approach_delta(session: Any, player: Any, topic: str, approach: str, si: Dict[str, Any]) -> Tuple[float, str]:
    pers = _personality(player)
    coach = _f(pers.get("coachability"), 55)
    ego = _f(pers.get("ego"), 50)
    comp = _f(pers.get("competitiveness"), 55)
    res = _f(pers.get("resilience"), 55)
    amb = _f(pers.get("ambition"), 55)
    age = _age(player)
    d = _ovr_delta(player)
    crowd_label = _depth_crowd(_user_team(session), player)[1] if _user_team(session) is not None else "competitive"
    noise = _rng(_pid(player), _day(session), topic, approach).uniform(-1.5, 1.5)
    delta, reply = 0.0, "Okay."
    if approach == "support":
        delta = 3.0 + (2.0 if age <= 19 else 0.0) + (55 - res) * 0.05
        reply = "That means a lot. Thanks for checking in."
    elif approach == "push":
        delta = (comp - 55) * 0.12 + (res - 55) * 0.08 - 1.0
        reply = "Yeah. You're right. I'll be better." if delta > 0 else "I'm working as hard as I can. That's not fair."
    elif approach == "plan":
        delta = 2.0 + (coach - 55) * 0.10
        reply = "That's what I wanted to hear. I'll follow it." if delta > 1.5 else "Okay. I'll look at it."
    elif approach == "praise":
        delta = 2.5 if d >= 0 else -1.5
        reply = "Appreciate it. I'm putting the work in." if d >= 0 else "I don't feel like I've earned that."
    elif approach == "role_ok":
        delta = 1.0 + (coach - 55) * 0.08 - max(0.0, ego - 62) * 0.12
        reply = "Fair. I'll make the most of it." if delta > 0 else "I think I'm better than that role."
    elif approach == "role_more":
        delta = 3.5 + max(0.0, amb - 55) * 0.05
        try:
            player._rights_talk_role_promise_day = _day(session)
        except Exception:
            pass
        reply = "That's huge. I'll hold you to it."
    elif approach == "path":
        delta = {"open": 6.0, "competitive": 2.5, "crowded": -2.0}.get(crowd_label, 2.0)
        reply = ("I can see it. That's a real shot." if delta > 3 else
                 "Maybe. There are a lot of guys ahead of me." if delta > 0 else
                 "With respect, I've looked at your depth chart.")
    elif approach == "commit":
        ready = _ovr(player) >= 70
        delta = (4.0 if ready else -1.5) - max(0.0, ego - 65) * 0.08 + (2.0 if si["interest"] >= 58 else -1.0)
        reply = "That feels good, knowing you want me." if delta > 0 else "I don't want to be rushed."
    elif approach == "respect":
        delta = 2.0 + (1.5 if si["leaning_no"] else 0.0)
        reply = "Thanks for not pushing. I'll keep you in the loop."
    elif approach == "back_choice":
        delta = 3.0
        try:
            if si["path"] in ("ncaa", "junior", "europe") and _ovr(player) < 72:
                player.willingness_to_sign = None if getattr(player, "willingness_to_sign", None) is False else getattr(player, "willingness_to_sign", None)
        except Exception:
            pass
        reply = "Means a lot that you trust me on this."
    elif approach == "go_pro":
        ready = _ovr(player) >= 72
        delta = 3.0 if ready else -2.0 - max(0.0, ego - 60) * 0.05
        if ready:
            try:
                player.willingness_to_sign = True
            except Exception:
                pass
        reply = "You know what, I think I'm ready." if ready else "I'm not there yet. I know my body."
    return round(delta + noise, 1), reply


def overview(session: Any, player: Any) -> Dict[str, Any]:
    si = signing_interest(session, player)
    line = _season_line(player)
    path = si["path"]
    topics = [{"id": t, "label": lbl} for t, lbl in TOPICS
              if not (t == "pro_or_stay" and path == "other")]
    if path == "ncaa":
        topics = [{**t, "label": "Turn pro or stay in school?"} if t["id"] == "pro_or_stay" else t for t in topics]
    log = dict(getattr(player, "_rights_talk_log", None) or {})
    day = _day(session)
    for t in topics:
        last = log.get(t["id"])
        t["cooldown_days"] = max(0, TALK_COOLDOWN_DAYS - (day - _i(last))) if last is not None else 0
    opener = _pick(("Hey! Didn't expect a call from the GM.", "Hi. Everything okay?", "Good to hear from you."),
                   _pid(player), day, "open") if si["interest"] >= WARNING_LINE else _pick(
        ("Oh. Hi.", "Yeah, I've got a minute.", "Didn't expect to hear from you."), _pid(player), day, "open")
    return {
        "ok": True,
        "player_id": _pid(player),
        "player_name": _name(player),
        "age": _age(player),
        "ovr": round(_ovr(player)),
        "ovr_delta_season": _ovr_delta(player),
        "position": _pos_bucket(player),
        "path": path,
        "season_line": line,
        "signing": si,
        "rights_expiry_year": getattr(player, "rights_expiry_year", None),
        "rights_signing_deadline": getattr(player, "rights_signing_deadline", None),
        "opener": opener,
        "topics": topics,
        "history": list(getattr(player, "_rights_talk_history", None) or [])[-6:],
    }


def talk(session: Any, player_id: str, topic: str = "", approach: str = "") -> Dict[str, Any]:
    player = _find(session, player_id)
    if player is None:
        return {"ok": False, "reason": "Prospect not found in your rights list."}
    if not topic:
        return overview(session, player)
    topic = str(topic)
    if topic not in dict(TOPICS):
        return {"ok": False, "reason": "Unknown topic."}
    si = signing_interest(session, player)
    answer = _answer(session, player, topic, si)
    if not approach:
        return {**overview(session, player), "topic": topic, "answer": answer,
                "approaches": [{"id": a, "label": l, "detail": d} for a, l, d in _APPROACHES.get(topic, ())]}
    if approach not in {a for a, _, _ in _APPROACHES.get(topic, ())}:
        return {"ok": False, "reason": "Unknown approach."}
    day = _day(session)
    log = dict(getattr(player, "_rights_talk_log", None) or {})
    last = log.get(topic)
    on_cd = last is not None and day - _i(last) < TALK_COOLDOWN_DAYS
    delta, reply = _approach_delta(session, player, topic, approach, si)
    if on_cd:
        delta = min(delta, 0.0) * 0.5  # he heard this two weeks ago: no new credit
        reply = "We talked about this already. " + reply
    before = si["interest"]
    rapport = _clip(_f(getattr(player, "_rights_talk_rapport", 0.0)) + delta, -18.0, 18.0)
    log[topic] = day
    hist = list(getattr(player, "_rights_talk_history", None) or [])
    hist.append({"day": day, "topic": topic, "approach": approach, "delta": delta})
    try:
        player._rights_talk_rapport = rapport
        player._rights_talk_log = log
        player._rights_talk_history = hist[-12:]
        rel = _f(getattr(player, "org_relationship", None), 0.55)
        player.org_relationship = round(max(0.0, min(1.0, rel + delta * 0.004)), 3)
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    after = signing_interest(session, player)
    msg = f"Signing interest {before:.0f} → {after['interest']:.0f}."
    if si["refuses"] and not after["refuses"]:
        msg = f"{_name(player)} is reconsidering: he's open to signing with you again. " + msg
    return {**overview(session, player), "topic": topic, "answer": answer, "reply": reply,
            "interest_before": before, "interest_after": after["interest"], "delta": round(after["interest"] - before, 1),
            "message": msg}
