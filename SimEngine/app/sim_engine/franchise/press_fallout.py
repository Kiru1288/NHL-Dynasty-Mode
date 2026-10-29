"""What a press answer actually sounds like, and what it does to people.

* ``gm_quote``       — the words the GM said, in his voice (2-3 variants per answer)
* ``react``          — the player hears it: an emotion, a belief shift, a reaction line
* ``room_fallout``   — teammates who weren't named still hear it (jealousy, anxiety)
* ``maybe_followup`` — a dodge gets a follow-up question from the same reporter
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.sim_engine.franchise import player_mind as mind

_log = logging.getLogger(__name__)

# response id -> quotes. {p} player, {rec} record.
_QUOTES: Dict[str, List[str]] = {
    "stay_course": ["We're not going to blow up a plan over a bad stretch. The process is right.",
                    "I understand the frustration at {rec}. We believe in what we're building."],
    "acknowledge": ["{rec} isn't good enough, and nobody in that room thinks it is.",
                    "We own this. It's on all of us, me included."],
    "changes": ["Everything is on the table. I won't sit still if this keeps up.",
                "If the results don't change, the people will. That's the business."],
    "credit_team": ["This is the room. They've bought in, and it shows every night.",
                    "The players deserve the credit. They've driven this."],
    "stay_humble": ["It's a good stretch. That's all it is. We haven't won anything yet.",
                    "Nobody's getting carried away in there, and I won't either."],
    "raise_bar": ["Good isn't the standard. We expect more, even now.",
                  "I like the record. I don't love some of the hockey. We can be better."],
    "push_hard": ["Make no mistake, the playoffs are the goal. Everything points at that.",
                  "We're playing meaningful games and we intend to keep playing them."],
    "one_game": ["Honestly? Tomorrow's game. That's as far as I'm looking.",
                 "One night at a time. The standings take care of themselves."],
    "honest_health": ["I'll be straight with you on health: we're banged up, and it matters.",
                      "Availability is the story right now. We need bodies back."],
    "compete": ["We're not tanking. Nobody in that room is playing for a draft pick.",
                "I'd never ask a player to lose. We're here to compete."],
    "honest_build": ["This year is about our young players. I'm not going to pretend otherwise.",
                     "We're building something. Some nights that's going to hurt."],
    "deflect_future": ["I'm not putting a label on this season.",
                       "Call it whatever you want. We'll keep working."],
    "deny": ["{p} isn't going anywhere. I'm not shopping him.",
             "Not a single call from me about {p}. He's part of this."],
    "neither_confirm": ["I don't talk about trades, and I'm not starting today.",
                        "You know I won't get into that."],
    "listen": ["Teams call about good players. {p} is a good player. I listen.",
               "My phone rings. I'd be bad at my job if I didn't pick it up."],
    "deflect": ["That stays between us and the coaching staff.",
                "Internal conversations stay internal."],
    "support_staff": ["Full confidence in our coaches. Full stop.",
                      "The staff has my backing. This isn't on them."],
    "support_player": ["{p} has my full support. I believe in him completely.",
                       "I'm in {p}'s corner. He'll come out the other side of this."],
    "no_comment": ["No comment.", "I'm not going to get into that today."],
    "optimistic": ["We think {p} is close. I'm hopeful we see him soon.",
                   "Good signs with {p}. I'm optimistic."],
    "no_timeline": ["No timeline. We'll know more when we know more.",
                    "I'm not putting a date on it."],
    "transparent": ["{p} is going to miss real time. The lineup has to adjust and it will.",
                    "Straight answer: it's significant. We'll fill the hole as a group."],
    "fair_talks": ["We're talking with {p}'s camp and it's been good. We want him here.",
                   "Conversations are ongoing and they're positive."],
    "focus_hockey": ["{p} is focused on hockey and so am I. The rest takes care of itself.",
                     "That's for his agent and me. Today's about the game."],
    "united": ["That room is together. I'd put it up against anyone's.",
               "Whatever you've heard, they're united."],
    "internal": ["We'll handle that inside the room.",
                 "Not something I'm airing in public."],
    "accountability": ["Some guys need to be better, and they know who they are.",
                       "Standards apply to everyone. Including our best players."],
    "praise": ["{p} has been outstanding. He's earned every bit of this.",
               "What {p} is doing doesn't happen by accident. Credit to him."],
    "temper": ["{p} is playing well. Let's not get carried away.",
               "It's a good stretch from {p}. Now do it for a season."],
    "credit_staff": ["Our coaches put {p} in spots to succeed, and he's taken them.",
                     "That's the staff's work as much as anyone's."],
    "address": ["It's a problem, and we've got a plan to fix it. I'll own that.",
                "Yes, there's an issue. Here's what we're doing about it."],
    "trust_process": ["Trust what we're doing. It's going to pay off.",
                      "Stick with us. The plan hasn't changed."],
    "compete_daily": ["You'll get effort every single night. That I can promise.",
                      "We'll compete. That part's not negotiable."],
    "clarify": ["Fair question. Let me be clearer than I was.",
                "Let me put it plainly this time."],
    "stand_by": ["I said what I said.", "My answer hasn't changed."],
    "shut_down": ["We're done with that one.", "Next question."],
}

# response id -> (emotion, intensity, belief changes) for the player named in the story
_REACTIONS: Dict[str, Tuple[str, float, Dict[str, float]]] = {
    "praise": ("pride", 38, {"gm_backs_me": 6}),
    "support_player": ("gratitude", 40, {"gm_backs_me": 9, "being_shopped": -6}),
    "temper": ("hurt", 16, {"gm_backs_me": -2}),
    "credit_staff": ("hurt", 10, {"coach_rates_me": 3}),
    "address": ("embarrassment", 26, {"gm_backs_me": -2}),
    "no_comment": ("hurt", 30, {"gm_backs_me": -7}),
    "internal": ("relief", 12, {}),
    "deny": ("relief", 30, {"being_shopped": -20, "club_wants_me_long": 5}),
    "neither_confirm": ("anxiety", 30, {"being_shopped": 12}),
    "listen": ("anxiety", 48, {"being_shopped": 28, "gm_backs_me": -6, "club_wants_me_long": -10}),
    "fair_talks": ("excitement", 22, {"club_wants_me_long": 12}),
    "focus_hockey": ("anxiety", 12, {"club_wants_me_long": -3}),
    "optimistic": ("relief", 14, {}),
    "transparent": ("relief", 10, {}),
    "no_timeline": ("anxiety", 8, {}),
    "deflect": ("anxiety", 12, {"gm_backs_me": -2}),
    "support_staff": ("hurt", 14, {"gm_backs_me": -4, "coach_rates_me": -2}),
    "clarify": ("relief", 16, {"gm_backs_me": 3, "being_shopped": -8}),
    "stand_by": ("anxiety", 10, {"being_shopped": 4}),
    "shut_down": ("anxiety", 18, {"being_shopped": 8, "gm_backs_me": -3}),
}

_REACTION_LINES: Dict[str, List[str]] = {
    "pride": ["{p} was asked about it after practice: “That means a lot coming from him.”",
              "{p} smiled when he heard it. “Nice to know the boss sees it.”"],
    "gratitude": ["{p}: “He didn't have to say that. I won't forget it.”",
                  "{p} texted the GM a thank-you within the hour."],
    "relief": ["{p} looked lighter at the morning skate.",
               "{p}: “Good. Now I can just play.”"],
    "excitement": ["{p}: “I want to be here. Glad they feel the same.”"],
    "hurt": ["{p} read it. Teammates say he went quiet.",
             "{p} declined to comment when reporters asked what he made of it."],
    "embarrassment": ["{p} kept his answers short with reporters the next day.",
                      "{p}: “I saw it. I get it. Nothing else to say.”"],
    "anxiety": ["{p}'s agent called the club that afternoon.",
                "{p}: “I just want to know where I stand.”"],
}

# Small feelings get small reactions; "went quiet" is for things that really land.
_MILD_LINES: Dict[str, List[str]] = {
    "hurt": ["{p} shrugged it off when asked, mostly.", "{p}: \u201cHe's entitled to his view.\u201d"],
    "anxiety": ["{p} asked a teammate what he made of it.", "{p} read it twice."],
    "relief": ["{p} nodded when it was read to him."],
    "pride": ["{p}: \u201cAppreciate it.\u201d"],
}

_FOLLOWUPS: Dict[str, str] = {
    "trade": "You didn't deny it. Should {p}'s teammates read anything into that?",
    "contract": "So is there a contract offer on the table for {p} or not?",
    "player": "That sounded like you're not fully behind {p}. Is that fair?",
    "coach": "That wasn't a yes. Is the coach's job safe?",
    "injury": "Is there something you're not telling us about {p}?",
    "room": "Players are talking. If it's fine, why not say so?",
    "record": "Fans want to hear more than that. What actually changes?",
    "direction": "Fans want to hear more than that. What actually changes?",
}

DODGES = ("no_comment", "deflect", "neither_confirm", "internal", "focus_hockey", "no_timeline", "deflect_future")


def _fill(text: str, entry: Dict[str, Any]) -> str:
    rec = str((entry.get("context") or {}).get("record") or "")
    return text.replace("{p}", str(entry.get("player_name") or "him")).replace("{rec}", rec or "this record")


def gm_quote(session: Any, entry: Dict[str, Any], question_id: str, response_id: str) -> str:
    pool = _QUOTES.get(response_id) or ["We'll leave it there."]
    rng = mind.mind_rng(session, "press_quote", entry.get("id"), question_id, response_id)
    return _fill(rng.choice(pool), entry)


def react(session: Any, entry: Dict[str, Any], question_id: str, response_id: str, quote: str) -> Optional[str]:
    """The named player hears the answer. Returns a reaction line for the coverage."""
    plid = str(entry.get("player_id") or "")
    ent = mind.entity_for(session, plid) if plid else None
    if not ent:
        return None
    spec = _REACTIONS.get(response_id)
    if response_id == "no_comment" and question_id in ("q_story_good",):
        spec = ("hurt", 14, {"gm_backs_me": -3})
    if not spec:
        return None
    kind, intensity, beliefs = spec
    cause = f"GM to the press: “{quote}”"
    # A player who already doubts the GM takes a cold answer harder, a warm one with salt.
    backs = float(mind.ensure_mind(ent)["beliefs"].get("gm_backs_me", 55))
    if kind in ("hurt", "anxiety", "embarrassment") and backs < 40:
        intensity *= 1.3
    elif kind in ("pride", "gratitude") and backs < 35:
        intensity *= 0.6
    mind.feel(session, ent, kind, intensity, cause)
    for key, delta in beliefs.items():
        mind.shift_belief(session, ent, key, delta, cause)
    rng = mind.mind_rng(session, "press_react", entry.get("id"), response_id)
    pool = _REACTION_LINES.get(kind) or ["{p} heard it."]
    if intensity < 18:
        pool = _MILD_LINES.get(kind) or pool
    return _fill(rng.choice(pool), entry)


def room_fallout(session: Any, entry: Dict[str, Any], question_id: str, response_id: str, quote: str) -> List[str]:
    """Teammates who weren't the subject still hear it."""
    utid = str(getattr(session, "user_team_id", "") or "")
    team = (getattr(session, "team_by_id", None) or {}).get(utid)
    ents = getattr(session, "universe_players", None) or {}
    if team is None:
        return []
    roster = [p for p in list(getattr(team, "roster", None) or []) if isinstance(ents.get(str(getattr(p, "id", ""))), dict)]
    notes: List[str] = []
    cause = f"GM to the press: “{quote}”"
    plid = str(entry.get("player_id") or "")
    if response_id in ("changes", "accountability", "raise_bar"):
        # Everyone wonders if it's about them; the ones already unsure feel it most.
        hit = 0
        for p in roster:
            ent = ents[str(p.id)]
            secure = float(mind.ensure_mind(ent)["beliefs"].get("role_secure", 55))
            if secure < 50:
                mind.feel(session, ent, "anxiety", 22 if response_id == "changes" else 12, cause, journal=hit < 2)
                mind.shift_belief(session, ent, "role_secure", -5, cause)
                hit += 1
        if hit:
            notes.append(f"{hit} players on the bubble took it personally.")
    elif response_id == "praise" and plid:
        # Praise one player, and a competitor at his position notices.
        named = next((p for p in roster if str(p.id) == plid), None)
        pos = str(getattr(named, "position", "") or "")[:1] if named is not None else ""
        rivals = [
            p for p in roster
            if str(p.id) != plid and str(getattr(p, "position", "") or "")[:1] == pos
            and float((ents[str(p.id)].get("state") or {}).get("role_satisfaction", 60) or 60) < 55
        ]
        if rivals:
            rival = rivals[mind.mind_rng(session, "press_rival", entry.get("id")).randrange(len(rivals))]
            ent = ents[str(rival.id)]
            mind.feel(session, ent, "jealousy", 20, f"GM singled out {entry.get('player_name')} in the press")
            notes.append(f"{ent.get('player_name')} noticed who got the credit.")
    elif response_id in ("support_staff",) and plid:
        notes.append("The coaches heard it. So did the player.")
    elif response_id in ("united", "credit_team", "compete_daily"):
        for p in roster[:23]:
            mind.feel(session, ents[str(p.id)], "pride", 8, cause, journal=False)
    return notes


def maybe_followup(session: Any, entry: Dict[str, Any], question: Dict[str, Any], response_id: str, topic: str) -> Optional[Dict[str, Any]]:
    """A dodge invites a follow-up from the same reporter (once per availability)."""
    if response_id not in DODGES or entry.get("followup_asked"):
        return None
    if str(question.get("id") or "").startswith("q_followup"):
        return None
    text = _FOLLOWUPS.get(topic) or _FOLLOWUPS["record"]
    if "{p}" in text and not entry.get("player_name"):
        text = _FOLLOWUPS["record"]
    from app.sim_engine.franchise.storyline_engine import _press_resp  # noqa: WPS433

    qid = "q_followup_trade" if topic == "trade" else "q_followup"
    entry["followup_asked"] = True
    return {
        "id": qid,
        "reporter_id": question.get("reporter_id"),
        "reporter_name": question.get("reporter_name"),
        "outlet": question.get("outlet") or "",
        "question": _fill(text, entry),
        "requires_triggers": [],
        "context_tags": ["follow_up"],
        "is_followup": True,
        "responses": [
            _press_resp(qid, "clarify", "Clarify", "Give a straighter answer. Calms the story.", "honest"),
            _press_resp(qid, "stand_by", "Stand by it", "Repeat yourself. The story keeps going.", "firm"),
            _press_resp(qid, "shut_down", "Shut it down", "End it here. The reporter won't forget.", "cold"),
        ],
    }
