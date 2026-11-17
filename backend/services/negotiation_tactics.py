"""Negotiation tactics: 30 creative moves you can make during contract talks.

Each tactic is a real lever on the contract engine, not flavour text:

* ``interest_delta`` — points on the Deal interest meter (summed with pitches / agent meetings
  in ``negotiation_meetings._interest_bonus_from_ledger``).
* ``ask_shift_pct`` — moves his asking price for your club (stacked in
  ``negotiation_meetings.demand_multiplier`` through ``player._tactic_ask_shift``).

How well a tactic lands = how much *this* player cares about what it sells x how well your
club can actually back it up (your roster, your cap, your market), plus a small seeded roll.
Teammate tactics use real players on your roster and they speak in the transcript.

Limits: each tactic once per player per window, and ``MAX_TACTICS`` moves per negotiation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import logging as _logging_swallow

_swallowed_log = _logging_swallow.getLogger(__name__)

MAX_TACTICS = 3

# (id, label, group, scope, one-line description). scope: "all" | "fa" | "own"
TACTICS: Tuple[Tuple[str, str, str, str, str], ...] = (
    # --- Your locker room
    ("captain_call", "Captain's phone call", "Locker room", "all", "Your captain calls him directly."),
    ("star_recruit", "Your star recruits him", "Locker room", "all", "Your best player makes the pitch."),
    ("linemate_pitch", "Pitch a linemate", "Locker room", "all", "Show him who he'd play beside."),
    ("countryman_dinner", "Countryman's dinner", "Locker room", "all", "A teammate from his country hosts dinner."),
    ("veteran_mentor", "Veteran mentor", "Locker room", "all", "A veteran promises to take him under his wing."),
    ("same_agent_client", "Same-agent teammate vouches", "Locker room", "all", "A teammate his agent also represents vouches for you."),
    ("locker_room_video", "Locker-room video", "Locker room", "all", "The whole room records a welcome video."),
    # --- Off the ice
    ("family_tour", "Family tour of the city", "Off the ice", "all", "Schools, neighbourhoods, the works."),
    ("owner_dinner", "Dinner with the owner", "Off the ice", "all", "Ownership shows how much they want him."),
    ("coach_call", "Head coach lays out his role", "Off the ice", "all", "Your coach walks through his usage."),
    ("legend_call", "A franchise legend calls", "Off the ice", "all", "A club icon tells him what it means to play here."),
    ("charity_pledge", "Back his charity", "Off the ice", "all", "The club pledges to support his foundation."),
    ("jersey_number", "Clear his number", "Off the ice", "all", "His number is waiting in the room."),
    ("facilities_tour", "Training-staff tour", "Off the ice", "all", "Sports science, skills coaches, recovery."),
    ("analytics_show", "Show the analytics plan", "Off the ice", "all", "Deployment charts built around his game."),
    ("fan_campaign", "Fan social campaign", "Off the ice", "all", "Rally the fan base online."),
    ("hometown_roots", "Play up his roots", "Off the ice", "all", "Sell coming home to play in his own country."),
    # --- Promises
    ("ice_time_promise", "Promise top-six minutes", "Promises", "all", "Guarantee a big role."),
    ("pp_promise", "Promise power-play time", "Promises", "all", "First-unit power play."),
    ("leadership_role", "Offer a letter", "Promises", "all", "An 'A' on his sweater."),
    ("trade_assurance", "Handshake: no trade", "Promises", "all", "Your word you won't move him."),
    ("front_load", "Front-load the money", "Promises", "all", "More of the money up front."),
    ("performance_bonus", "Add performance bonuses", "Promises", "all", "Lower base, upside if he produces."),
    # --- Hardball
    ("deadline", "Set a deadline", "Hardball", "all", "The offer comes off the table Friday."),
    ("walk_away", "Walk away from the table", "Hardball", "all", "Bluff: you're ready to move on."),
    ("alternative_target", "Mention your other target", "Hardball", "fa", "Let slip you're talking to a similar player."),
    ("comparable_contracts", "Show comparable contracts", "Hardball", "all", "Put the comps on the table."),
    ("cap_crunch", "Plead a cap crunch", "Hardball", "all", "Show him there's no more room."),
    # --- Wildcards
    ("media_blitz", "Go public with your interest", "Wildcard", "all", "Tell the press how badly you want him."),
    ("mascot_stunt", "Send the mascot", "Wildcard", "all", "The mascot delivers a gift basket."),
)

_LABEL = {t[0]: t[1] for t in TACTICS}


# --------------------------------------------------------------------------- helpers


def _ce():
    from services import contract_economy as ce

    return ce


def _roster(session: Any) -> List[Any]:
    from services.negotiation_meetings import _user_team

    team = _user_team(session)
    return [p for p in list(getattr(team, "roster", None) or [])] if team is not None else []


def _name(p: Any) -> str:
    try:
        return str(_ce()._player_name(p))
    except Exception:
        return "A teammate"


def _ovr(p: Any) -> float:
    try:
        return float(_ce()._player_ovr(p))
    except Exception:
        return 70.0


def _age(p: Any) -> int:
    try:
        return int(_ce()._player_age(p) or 27)
    except Exception:
        return 27


def _pos(p: Any) -> str:
    try:
        return str(_ce()._position_bucket(p))
    except Exception:
        return ""


def _country(p: Any) -> str:
    ce = _ce()
    ident = ce._get(p, "identity", None)
    raw = str(ce._get(ident, "birth_country", "") or ce._get(p, "birth_country", "") or ce._get(p, "nationality", "") or "")
    raw = raw.strip().upper()
    return {"CANADA": "CAN", "CA": "CAN", "USA": "USA", "US": "USA", "UNITED STATES": "USA", "SWEDEN": "SWE", "SE": "SWE",
            "FINLAND": "FIN", "FI": "FIN", "RUSSIA": "RUS", "RU": "RUS", "CZECHIA": "CZE", "CZECH REPUBLIC": "CZE", "CZ": "CZE",
            "SLOVAKIA": "SVK", "SK": "SVK", "GERMANY": "GER", "DE": "GER", "SWITZERLAND": "SUI", "CH": "SUI"}.get(raw, raw)


def _pid(p: Any) -> str:
    try:
        return str(_ce()._player_id(p))
    except Exception:
        return ""


def _morale(session: Any, p: Any) -> float:
    ent = (getattr(session, "universe_players", None) or {}).get(_pid(p)) or {}
    try:
        return float((ent.get("state") or {}).get("morale", 55) or 55)
    except (TypeError, ValueError):
        return 55.0


def _captain(session: Any, target: Any) -> Optional[Any]:
    from services.negotiation_meetings import _user_team

    team = _user_team(session)
    cid = str(getattr(team, "captain_id", getattr(team, "captain", "")) or "") if team is not None else ""
    roster = [p for p in _roster(session) if _pid(p) != _pid(target)]
    if cid:
        for p in roster:
            if _pid(p) == cid:
                return p
    vets = sorted((p for p in roster if _age(p) >= 26), key=lambda p: -(_ovr(p) + _age(p) * 0.4))
    return vets[0] if vets else (roster[0] if roster else None)


def _star(session: Any, target: Any) -> Optional[Any]:
    roster = [p for p in _roster(session) if _pid(p) != _pid(target)]
    return max(roster, key=_ovr) if roster else None


def _fwd(pos: str) -> bool:
    return pos in ("C", "LW", "RW", "F", "W")


def _linemate(session: Any, target: Any) -> Optional[Any]:
    tpos = _pos(target)
    roster = [p for p in _roster(session) if _pid(p) != _pid(target)]
    if tpos == "G":
        pool = [p for p in roster if _pos(p) not in ("G",) and not _fwd(_pos(p))]
    elif tpos == "C":
        pool = [p for p in roster if _pos(p) in ("LW", "RW", "W")]
    elif _fwd(tpos):
        pool = [p for p in roster if _pos(p) == "C"] or [p for p in roster if _fwd(_pos(p))]
    else:
        pool = [p for p in roster if not _fwd(_pos(p)) and _pos(p) != "G"]
    return max(pool, key=_ovr) if pool else None


def _countryman(session: Any, target: Any) -> Optional[Any]:
    c = _country(target)
    if not c:
        return None
    pool = [p for p in _roster(session) if _pid(p) != _pid(target) and _country(p) == c]
    return max(pool, key=_ovr) if pool else None


def _veteran(session: Any, target: Any) -> Optional[Any]:
    pool = [p for p in _roster(session) if _pid(p) != _pid(target) and _age(p) >= 31]
    same = [p for p in pool if _pos(p) == _pos(target) or (_fwd(_pos(p)) and _fwd(_pos(target)))]
    pool = same or pool
    return max(pool, key=_ovr) if pool else None


def _same_agent(session: Any, target: Any) -> Optional[Any]:
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent

        aid = str((ensure_player_agent(target, session) or {}).get("id") or "")
        if not aid:
            return None
        for p in _roster(session):
            if _pid(p) == _pid(target):
                continue
            if str((ensure_player_agent(p, session) or {}).get("id") or "") == aid:
                return p
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return None


def _team_country(session: Any) -> str:
    from services.negotiation_meetings import _user_team

    try:
        return "CAN" if _ce()._team_is_canadian(_user_team(session)) else "USA"
    except Exception:
        return ""


def _cap_space(session: Any) -> float:
    try:
        from services.league_governance import _cap_space as gov_cap_space
        from services.negotiation_meetings import _user_team

        return float(gov_cap_space(session, _user_team(session)))
    except Exception:
        return 5.0


def _fan_engagement(session: Any) -> float:
    try:
        from services.franchise_sim import _ensure_team_fan_profile
        from services.negotiation_meetings import _uid

        prof = _ensure_team_fan_profile(session, _uid(session)) or {}
        return float(prof.get("engagement") or prof.get("fan_engagement") or 55) / 100.0
    except Exception:
        return 0.55


def _room_mood(session: Any) -> float:
    roster = _roster(session)
    if not roster:
        return 0.55
    return sum(_morale(session, p) for p in roster) / len(roster) / 100.0


def _agent_style(session: Any, target: Any) -> str:
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent

        return str((ensure_player_agent(target, session) or {}).get("style") or "")
    except Exception:
        return ""


def _agent_name(session: Any, target: Any) -> str:
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent

        return str((ensure_player_agent(target, session) or {}).get("name") or "His agent")
    except Exception:
        return "His agent"


def _first(name: str) -> str:
    return (str(name or "").split() or ["He"])[0]


# --------------------------------------------------------------------------- tactic table
#
# Each spec returns a dict:
#   available: bool, why_not: str, speaker: (who, name) for the move itself,
#   care: 0..1 how much he cares, strength: 0..1 how well you back it up, fact: str,
#   ask_shift: float intensity (negative = cheaper; see _price_swing_pct), risk: float (how hard a miss backfires)


def _spec(session: Any, player: Any, where: str, tid: str, traits: Dict[str, float]) -> Dict[str, Any]:
    from services.negotiation_meetings import _projected_slot

    age = traits.get("age", 27.0)
    ovr = _ovr(player)
    pos = _pos(player)
    s: Dict[str, Any] = {"available": True, "why_not": "", "care": 0.5, "strength": 0.5, "fact": "", "ask_shift": 0.0, "risk": 1.0,
                         "speaker": ("you", "You"), "helper": None}

    def need(p: Optional[Any], why: str) -> Optional[Any]:
        if p is None:
            s["available"] = False
            s["why_not"] = why
        return p

    if tid == "captain_call":
        c = need(_captain(session, player), "You need a captain or veteran on the roster.")
        if c is not None:
            s.update(helper=c, speaker=("teammate", _name(c)), care=0.35 + traits["winning"] * 0.5,
                     strength=min(1.0, (_ovr(c) - 70) / 22 + 0.15), fact=f"{_name(c)} · {int(_ovr(c))} OVR")
    elif tid == "star_recruit":
        st = need(_star(session, player), "No one on your roster to make the call.")
        if st is not None:
            s.update(helper=st, speaker=("teammate", _name(st)), care=0.3 + traits["winning"] * 0.6,
                     strength=max(0.0, min(1.0, (_ovr(st) - 76) / 14)), fact=f"{_name(st)} · {int(_ovr(st))} OVR")
    elif tid == "linemate_pitch":
        lm = need(_linemate(session, player), "No obvious linemate or partner on your roster.")
        if lm is not None:
            s.update(helper=lm, speaker=("teammate", _name(lm)), care=0.3 + traits["role"] * 0.4 + traits["winning"] * 0.2,
                     strength=max(0.0, min(1.0, (_ovr(lm) - 72) / 16)), fact=f"Would play with {_name(lm)} ({int(_ovr(lm))} OVR)")
    elif tid == "countryman_dinner":
        cm = need(_countryman(session, player), "No teammate from his country.")
        if cm is not None:
            s.update(helper=cm, speaker=("teammate", _name(cm)), care=0.35 + traits["market"] * 0.5,
                     strength=0.7 + (0.2 if age <= 24 else 0.0), fact=f"{_name(cm)} · {_country(cm)}")
    elif tid == "veteran_mentor":
        vt = need(_veteran(session, player), "No veteran (31+) on your roster.")
        if vt is not None and age > 25:
            s.update(available=False, why_not="Only for young players (25 and under).")
        elif vt is not None:
            s.update(helper=vt, speaker=("teammate", _name(vt)), care=0.3 + traits["development"] * 0.6,
                     strength=max(0.2, min(1.0, (_ovr(vt) - 70) / 18)), fact=f"{_name(vt)} · age {_age(vt)}")
    elif tid == "same_agent_client":
        sa = need(_same_agent(session, player), "None of your players share his agent.")
        if sa is not None:
            s.update(helper=sa, speaker=("teammate", _name(sa)), care=0.6, strength=min(1.0, 0.45 + _morale(session, sa) / 120),
                     fact=f"{_name(sa)} · morale {int(_morale(session, sa))}")
    elif tid == "locker_room_video":
        mood = _room_mood(session)
        s.update(care=0.35 + traits["market"] * 0.3, strength=mood, fact=f"Room morale {int(mood * 100)}", risk=0.6)
    elif tid == "family_tour":
        s.update(care=0.2 + traits["market"] * 0.4 + (0.3 if age >= 27 else 0.0), strength=_fan_engagement(session) * 0.5 + 0.35,
                 fact="Family man" if age >= 27 else "Young and single")
    elif tid == "owner_dinner":
        s.update(care=0.3 + traits["security"] * 0.3 + traits["money"] * 0.2, strength=0.6, fact="Ownership commitment", ask_shift=-3.0)
    elif tid == "coach_call":
        label, score = _projected_slot(session, player)
        s.update(speaker=("teammate", "Head coach"), care=0.3 + traits["role"] * 0.6, strength=score, fact=f"Projects as: {label}")
    elif tid == "legend_call":
        s.update(speaker=("teammate", "Franchise legend"), care=0.25 + traits["loyalty"] * 0.35 + traits["market"] * 0.25,
                 strength=_fan_engagement(session), fact="Club history and fan engagement")
    elif tid == "charity_pledge":
        s.update(care=0.3 + traits["loyalty"] * 0.4, strength=0.7, fact="Community commitment", risk=0.4)
    elif tid == "jersey_number":
        s.update(care=0.4, strength=0.8, fact="Cosmetic, but personal", risk=0.3)
    elif tid == "facilities_tour":
        s.update(care=0.25 + traits["development"] * 0.6 + (0.15 if age >= 31 else 0.0), strength=0.65,
                 fact="Recovery staff for veterans" if age >= 31 else "Skills staff for his development")
    elif tid == "analytics_show":
        label, score = _projected_slot(session, player)
        s.update(care=0.25 + traits["role"] * 0.4 + traits["development"] * 0.2, strength=0.4 + score * 0.5, fact=f"Deployment: {label}")
    elif tid == "fan_campaign":
        eng = _fan_engagement(session)
        s.update(care=0.25 + traits["market"] * 0.5, strength=eng, fact=f"Fan engagement {int(eng * 100)}", risk=1.4)
    elif tid == "hometown_roots":
        tc, pc = _team_country(session), _country(player)
        if not pc or pc != tc:
            s.update(available=False, why_not="Only when he'd be playing in his home country.")
        else:
            s.update(care=0.4 + traits["loyalty"] * 0.3 + traits["market"] * 0.3, strength=0.75, fact=f"Born in {pc}")
    elif tid == "ice_time_promise":
        label, score = _projected_slot(session, player)
        # Promising what he already gets is weak; promising a jump he can't get is a stretch that can sour him.
        s.update(care=0.25 + traits["role"] * 0.7, strength=0.85 if score >= 0.8 else 0.6 if score >= 0.45 else 0.3,
                 fact=f"Currently projects: {label}", risk=1.3)
    elif tid == "pp_promise":
        off = _fwd(pos) or pos in ("D", "LD", "RD")
        s.update(care=0.25 + traits["role"] * 0.4 + traits["money"] * 0.2, strength=0.8 if (ovr >= 80 and off) else 0.45,
                 fact="Power-play numbers drive his next contract")
    elif tid == "leadership_role":
        if age < 26:
            s.update(available=False, why_not="He's too young for a letter (26+).")
        else:
            s.update(care=0.3 + traits["winning"] * 0.3 + traits["loyalty"] * 0.3, strength=min(1.0, (ovr - 70) / 18 + 0.2), fact=f"Age {int(age)}, {int(ovr)} OVR")
    elif tid == "trade_assurance":
        s.update(care=0.2 + traits["security"] * 0.7, strength=0.7, fact="A handshake, not a clause", ask_shift=-2.0)
    elif tid == "front_load":
        s.update(care=0.2 + traits["money"] * 0.7, strength=0.8, fact="Cash up front, same cap hit", ask_shift=-3.0)
    elif tid == "performance_bonus":
        s.update(care=0.2 + traits["money"] * 0.3 + (0.3 if age >= 33 else 0.0), strength=0.65 if age >= 33 else 0.45,
                 fact="Bonus-eligible veteran" if age >= 33 else "Bonuses are limited for him", ask_shift=-6.0)
    elif tid == "deadline":
        thin = 1.0 if where == "own" else max(0.0, min(1.0, (84 - ovr) / 14))
        s.update(speaker=("you", "You"), care=0.4, strength=thin, fact="Thin market" if thin >= 0.5 else "Plenty of suitors",
                 ask_shift=-10.0 * (thin - 0.4) * 2, risk=1.5)
    elif tid == "walk_away":
        thin = max(0.0, min(1.0, (86 - ovr) / 14))
        s.update(care=0.4, strength=thin, fact="Few teams can match you" if thin >= 0.5 else "Others will happily sign him",
                 ask_shift=-14.0 * (thin - 0.45) * 2, risk=2.0)
    elif tid == "alternative_target":
        if where != "fa":
            s.update(available=False, why_not="Free agents only.")
        else:
            comp = _comparable_fa(session, player)
            s.update(care=0.4, strength=0.75 if comp else 0.2, fact=f"Comparable: {comp}" if comp else "No real alternative on the market",
                     ask_shift=-9.0 if comp else 6.0, risk=1.5)
    elif tid == "comparable_contracts":
        gap = _ask_vs_market(session, player, where)
        s.update(care=0.4, strength=max(0.0, min(1.0, 0.5 + gap * 3)),
                 fact=f"Ask is {gap * 100:+.0f}% vs market" if gap else "Ask is at market", ask_shift=-12.0 * max(0.0, gap) * 4, risk=0.8)
    elif tid == "cap_crunch":
        space = _cap_space(session)
        tight = max(0.0, min(1.0, (8.0 - space) / 8.0))
        s.update(care=0.4, strength=tight, fact=f"${space:.1f}M in cap space", ask_shift=-10.0 * (tight - 0.4) * 2, risk=1.3)
    elif tid == "media_blitz":
        style = _agent_style(session, player)
        s.update(care=0.3 + traits["market"] * 0.4, strength=0.7, fact=f"Agent style: {style or 'unknown'}",
                 ask_shift=8.0 if style in ("media_savvy", "leaker", "leverage") else 2.0, risk=1.0)
    elif tid == "mascot_stunt":
        s.update(speaker=("teammate", "The mascot"), care=0.35, strength=0.55, fact="Nobody knows how this goes", risk=1.2)
    return s


def _comparable_fa(session: Any, player: Any) -> str:
    from services.negotiation_meetings import _league

    league = _league(session)
    ovr, pos, pid = _ovr(player), _pos(player), _pid(player)
    fwd = _fwd(pos)
    for p in list(getattr(league, "free_agents", None) or []):
        if _pid(p) == pid:
            continue
        same = (_fwd(_pos(p)) == fwd) and (_pos(p) == "G") == (pos == "G")
        if same and abs(_ovr(p) - ovr) <= 2:
            return _name(p)
    return ""


def _ask_vs_market(session: Any, player: Any, where: str) -> float:
    from services.negotiation_meetings import _league, _user_team

    try:
        row = _ce().compute_player_demand(player, _user_team(session), _league(session), context="re_sign" if where == "own" else "ufa")
        want = float(row.get("want_aav_m") or 0)
        market = float(row.get("market_value_m") or 0)
        return (want - market) / market if market > 0 else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- dialogue

def _lines(tid: str, ok: bool, big: bool, player_name: str, helper_name: str, agent: str, fact: str) -> Tuple[str, str, bool]:
    f = _first(player_name)
    h = helper_name or "your guy"
    say = {
        "captain_call": (f"Hey {f}, it's {_first(h)}. We've got something real going here. I want you on my team.",
                         f"{f} called {_first(h)} back the same night. He's in.", f"{f} thanked {_first(h)} for the call. Polite, nothing more."),
        "star_recruit": (f"{f}, you and me on the same line? Come on. Let's go win something.",
                         f"{f}: \"When {_first(h)} calls, you listen.\"", f"{f} respects {_first(h)}, but he's not choosing a team for one guy."),
        "linemate_pitch": (f"I've watched your tape. You'd fit what I do. Think about it.",
                           f"{f} is already picturing the chemistry with {_first(h)}.", f"{f} doesn't see the fit with {_first(h)}."),
        "countryman_dinner": (f"Come over for dinner. My wife makes the real food from home. You'll feel at home here.",
                              f"Two hours, three courses, zero hockey talk. {f} loved it.", f"Nice dinner. {f} still wants to see the numbers."),
        "veteran_mentor": (f"I wish someone had looked out for me at your age. I'll be that guy for you.",
                           f"{f} lit up. He wants to learn from {_first(h)}.", f"{f} thinks he's past needing a mentor."),
        "same_agent_client": (f"Ask {agent}. They treat us right here. No drama.", f"{agent}: \"If {_first(h)} says it's good, it's good.\"",
                              f"{agent}: \"{_first(h)}'s situation is different.\""),
        "locker_room_video": ("The guys put together a two-minute video. Half of it is the backup goalie dancing.",
                              f"{f} watched it three times.", f"{f} laughed, but it didn't change much."),
        "family_tour": ("We set up a day: schools, neighbourhoods, the rink, the waterfront.", f"{f}'s family loved it here.", "The tour was fine. Nothing jumped out."),
        "owner_dinner": ("Ownership flew in to tell him personally how much they want him.", f"{f} felt wanted. His camp softened on price.",
                         f"{agent}: \"Dinner's nice. Money's nicer.\""),
        "coach_call": (f"Here's your usage: {fact.split(': ', 1)[-1].lower()}, plus key minutes late in games.", f"{f} likes the plan.",
                       f"{f} expected a bigger role than that."),
        "legend_call": (f"Playing here changed my life. This town takes care of its players.", f"{f} was honoured by the call.", f"{f} appreciated it, but he's focused on the deal."),
        "charity_pledge": ("The club will match his foundation's fundraising for the length of the deal.", f"{f} was genuinely moved.", f"{f} appreciated it, but business is business."),
        "jersey_number": ("His number's already hanging in a stall with his name on it.", f"{f} sent a photo of it to his family.", f"{f} shrugged. It's just a number."),
        "facilities_tour": ("Full tour: skills coaches, sports science, recovery suite.", f"{f} was impressed by the staff.", f"{f} has seen better setups."),
        "analytics_show": ("Your analysts walk through charts built around his game.", f"{f} geeked out over the deployment plan.", f"{f} glazed over halfway through."),
        "fan_campaign": ("The fan base starts a campaign: #Bring" + f.replace(" ", "") + "Here.", f"{f} saw the posts. He noticed.",
                         "The campaign fizzled. A few dozen posts and a lot of bots."),
        "hometown_roots": (f"Come home, {f}. Play in front of your people.", f"{f} has always wanted to play at home.", f"{f} doesn't care where the rink is."),
        "ice_time_promise": ("You promise him top-six minutes and a real role.", f"{f} took you at your word.", f"{agent}: \"We've heard that before. Put it in the lineup.\""),
        "pp_promise": ("First power-play unit is his.", f"{f} loves the idea of running the power play.", f"{f} isn't sold on the power-play pitch."),
        "leadership_role": ("You tell him there's an 'A' waiting for him.", f"{f} was honoured.", f"{f} would rather earn it."),
        "trade_assurance": ("You give him your word he won't be moved.", f"{f} trusts you. His camp relaxed on the ask.", f"{agent}: \"A handshake isn't a clause.\""),
        "front_load": ("You offer to front-load the cash.", f"{agent} liked the structure and came down a bit.", f"{agent}: \"Structure's fine. The number's the problem.\""),
        "performance_bonus": ("You suggest a lower base with performance bonuses on top.", f"{agent} took the bonus structure.", f"{agent}: \"He's not betting on bonuses.\""),
        "deadline": ("You tell his camp the offer comes off the table Friday.", f"{agent} came back before Friday with a lower number.",
                     f"{agent}: \"Then pull it. Other teams are calling.\""),
        "walk_away": ("You close the folder and stand up.", f"{agent}: \"Sit down. Let's figure this out.\" The ask dropped.",
                      f"{agent} let you walk. The ask went up."),
        "alternative_target": ("You let it slip you're also talking to someone similar.", f"{agent} got nervous and came off the number.",
                               f"{agent}: \"Go sign him then.\" They called your bluff."),
        "comparable_contracts": ("You lay out comparable contracts around the league.", f"{agent} couldn't argue with the comps.", f"{agent}: \"Those comps don't fit him.\""),
        "cap_crunch": ("You show them your cap sheet. There's no more room.", f"{agent} saw the books and came down.", f"{agent}: \"You've got room. Don't insult us.\""),
        "media_blitz": ("You tell reporters he's your number-one target.", f"{f} loved the attention.", f"{agent} used it: \"Everyone knows how badly you want him.\""),
        "mascot_stunt": ("The mascot shows up at his door with a gift basket and a sign.", f"{f} posted a video. He's laughing.", f"{f} didn't know what to make of it."),
    }.get(tid, ("You make your move.", "It landed.", "It didn't land."))
    opener, win, miss = say
    return opener, (win if ok else miss), big


# --------------------------------------------------------------------------- public API


def tactic_options(session: Any, player: Any, where: str, traits: Dict[str, float], led: Dict[str, Any]) -> Dict[str, Any]:
    used = dict(led.get("tactics") or {})
    left = max(0, MAX_TACTICS - len(used))
    opts = []
    for tid, label, group, scope, desc in TACTICS:
        if scope == "fa" and where != "fa":
            continue
        if scope == "own" and where != "own":
            continue
        try:
            spec = _spec(session, player, where, tid, traits)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
            continue
        fit = spec["care"] * spec["strength"]
        opts.append({
            "id": tid, "label": label, "group": group, "line": desc,
            "available": bool(spec["available"]) and tid not in used and left > 0,
            "why_not": spec["why_not"] if not spec["available"] else ("Already used" if tid in used else ("No moves left" if left <= 0 else "")),
            "fact": spec["fact"],
            "helper": _name(spec["helper"]) if spec.get("helper") is not None else "",
            "read": "Strong fit" if fit >= 0.42 else "Worth a try" if fit >= 0.24 else "Long shot",
            "hardball": group == "Hardball",
            "result": used.get(tid),
        })
    return {"max": MAX_TACTICS, "left": left, "used": len(used), "options": opts}


def run_tactic(session: Any, player: Any, where: str, tid: str) -> Dict[str, Any]:
    from services.negotiation_meetings import (
        _contract_expiry, _current_ask, _ledger, _log, _price_swing_pct, _rng, _traits, _uid, _window,
    )

    if tid not in _LABEL:
        return {"ok": False, "reason": "Unknown tactic"}
    pid = _pid(player)
    led = _ledger(session, pid)
    used = led.setdefault("tactics", {})
    if tid in used:
        return {"ok": False, "reason": "You've already tried that with him."}
    if len(used) >= MAX_TACTICS:
        return {"ok": False, "reason": f"You've used all {MAX_TACTICS} moves in this negotiation."}
    traits = _traits(session, player)
    spec = _spec(session, player, where, tid, traits)
    if not spec["available"]:
        return {"ok": False, "reason": spec["why_not"] or "Not available."}

    rng = _rng("tactic", pid, _window(session), tid)
    roll = rng.uniform(-0.12, 0.12)
    care = max(0.0, min(1.0, float(spec["care"])))
    strength = max(0.0, min(1.0, float(spec["strength"])))
    score = care * strength + roll
    group = dict((t[0], t[2]) for t in TACTICS)[tid]
    ok = score >= 0.24
    big = score >= 0.45
    if ok:
        delta = 2.0 + (score - 0.24) * 22.0
    else:
        delta = -(0.24 - score) * 14.0 * float(spec["risk"])
    if not ok and group in ("Locker room", "Off the ice"):
        # A friendly gesture that falls flat costs a little, not a lot.
        delta = max(-3.0, delta)
    hardball = group == "Hardball"
    if hardball:
        # Hardball works on the price, not the love: a win barely moves interest, a miss stings.
        delta = 0.5 if ok else delta * 1.2
    delta = round(max(-9.0, min(9.0, delta)), 1)

    intensity = float(spec["ask_shift"] or 0.0)
    if intensity:
        if not ok and intensity < 0:
            intensity = abs(intensity) * 0.6  # a failed squeeze pushes the ask up
        intensity *= (1.2 if big else 1.0)
    ask_before = _current_ask(session, player, where) if intensity else 0.0
    pct = _price_swing_pct(intensity * 0.6, ask_before) if intensity else 0.0
    if pct:
        prev = getattr(player, "_tactic_ask_shift", None)
        base = float(prev.get("pct") or 0.0) if isinstance(prev, dict) and str(prev.get("team_id")) == _uid(session) and prev.get("expiry") == _contract_expiry(player) else 0.0
        try:
            setattr(player, "_tactic_ask_shift", {"team_id": _uid(session), "pct": round(max(-25.0, min(25.0, base + pct)), 1),
                                                   "window": _window(session), "expiry": _contract_expiry(player)})
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    ask_after = _current_ask(session, player, where) if pct else ask_before
    moved = round(ask_after - ask_before, 2) if (ask_before and ask_after) else 0.0

    name = _name(player)
    helper = _name(spec["helper"]) if spec.get("helper") is not None else ""
    agent = _agent_name(session, player)
    opener, outcome, _big = _lines(tid, ok, big, name, helper, agent, spec["fact"])
    who, speaker = spec["speaker"]
    transcript: List[Dict[str, str]] = []
    if who == "teammate":
        transcript.append({"who": "note", "text": f"{_LABEL[tid]}: {speaker} makes the call."})
        transcript.append({"who": "teammate", "name": speaker, "text": opener})
    else:
        transcript.append({"who": "you", "name": "You", "text": opener})
    if outcome.startswith(f"{agent}: \""):
        # '{agent}: "speech" narration' -> agent bubble + narration note
        body = outcome[len(agent) + 3:]
        speech, _, rest = body.partition('"')
        transcript.append({"who": "agent", "name": agent, "text": speech.strip()})
        if rest.strip():
            transcript.append({"who": "note", "text": rest.strip()})
    elif outcome.startswith(f"{agent} "):
        transcript.append({"who": "note", "text": outcome})
    else:
        transcript.append({"who": "note", "text": outcome})
    tail = []
    if delta:
        tail.append(f"interest {delta:+.1f}")
    if moved:
        tail.append(f"ask {'+' if moved > 0 else '−'}${abs(moved):.2f}M")
    if tail:
        transcript.append({"who": "note", "text": " · ".join(tail)})

    used[tid] = {"ok": ok, "interest_delta": delta, "ask_shift_pct": pct, "ask_shift_m": moved, "line": outcome, "fact": spec["fact"]}
    _log(led, f"Tactic — {_LABEL[tid]}: {'landed' if ok else 'missed'} ({delta:+.1f})")
    return {
        "ok": True, "kind": "tactic", "tactic": tid, "landed": ok, "interest_delta": delta, "ask_shift_m": moved,
        "message": outcome, "transcript": transcript, "moves_left": max(0, MAX_TACTICS - len(used)),
    }
