"""Meeting conversations with stakes.

This layer sits between the meeting templates (what a meeting is about and what each
answer *does*) and the player. It decides how a specific player hears a specific answer:

* **Who he is to the club.** A franchise player with an ego is a minefield; a fourth-liner
  can be told to earn it. Stature, ego, volatility, professionalism, broken promises,
  contract leverage and the club's record all feed how sensitive he is.
* **How the room feels.** Every meeting carries a tension level (his stress) and a GM
  pressure level (what's riding on it for you). Answers move tension; at 100 he walks out.
* **What you can say.** The opening options are re-cut for the player: a star gets
  "remind him what he means here" instead of "earn it or sit"; a depth player can be told
  plainly. The second beat depends on how he took the first: if he bristles you choose
  whether to lower the temperature or hold your ground, each with a real cost.
* **Suspense.** Exact effects stay hidden until the meeting ends. Choices carry a read
  ("He'll respect it", "Risky with him", "He might walk"), not numbers. Lines come with
  stage directions so the client can play the conversation out beat by beat.

Everything here is deterministic per meeting + choice, so a reload replays the same scene.
"""

from __future__ import annotations

import copy
import random
import zlib
from typing import Any, Dict, List, Optional, Tuple

# Keys where a smaller number is the good outcome.
_LOWER_IS_BETTER = {"tension", "media_stress", "personal_stress", "grievance", "friction", "state.media_stress",
                    "state.personal_stress", "state.grievance"}
_SKIP_SCALE = {"due_games", "success_readiness", "games", "days", "duration_games"}

STATURE_LABELS = {
    "franchise": "Franchise player",
    "core": "Core player",
    "regular": "Regular",
    "depth": "Depth player",
    "prospect": "Young player",
}

STATURE_SENSITIVITY = {"franchise": 1.6, "core": 1.3, "regular": 1.0, "depth": 0.7, "prospect": 0.85}

REACTION_ORDER = ["boils_over", "bristles", "guarded", "receptive", "opens_up"]
REACTION_LABELS = {
    "opens_up": "He opened up",
    "receptive": "He's listening",
    "guarded": "He's guarded",
    "bristles": "He bristled",
    "boils_over": "He's boiling",
    "walked_out": "He walked out",
}
# (positive multiplier, negative multiplier, tension change)
REACTION_EFFECT = {
    "opens_up": (1.25, 0.6, -20),
    "receptive": (1.0, 0.85, -10),
    "guarded": (0.75, 1.0, 5),
    "bristles": (0.5, 1.25, 15),
    "boils_over": (0.3, 1.5, 28),
}


def _rng(*parts: Any) -> random.Random:
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")))


def _f(v: Any, d: float = 50.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def _first(name: str) -> str:
    return str(name or "He").split(" ")[0]


def _last(name: str) -> str:
    bits = str(name or "").split(" ")
    return bits[-1] if bits else "him"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _team_rank(session: Any, player_id: str) -> Tuple[int, int]:
    """(rank by OVR on the user's NHL roster, roster size)."""
    utid = str(getattr(session, "user_team_id", "") or "")
    team = (getattr(session, "team_by_id", None) or {}).get(utid)
    roster = list(getattr(team, "roster", None) or [])

    def _ovr(p: Any) -> float:
        fn = getattr(p, "ovr", None)
        try:
            v = float(fn() if callable(fn) else fn or 0)
        except Exception:
            v = 0.0
        return v * 99.0 if v <= 1.5 else v

    ranked = sorted(roster, key=_ovr, reverse=True)
    for i, p in enumerate(ranked):
        if str(getattr(p, "id", "")) == str(player_id):
            return i + 1, len(ranked)
    return 0, len(ranked)


def _team_points_pct(session: Any) -> Optional[float]:
    utid = str(getattr(session, "user_team_id", "") or "")
    rec = (getattr(getattr(session, "standings", None), "records", None) or {})
    rr = rec.get(utid) if isinstance(rec, dict) else None
    if rr is None:
        return None
    w = int(getattr(rr, "wins", 0) or 0)
    l = int(getattr(rr, "losses", 0) or 0)
    o = int(getattr(rr, "ot_losses", 0) or getattr(rr, "otl", 0) or 0)
    gp = w + l + o
    if gp < 5:
        return None
    return (2 * w + o) / (2 * gp)


def build_context(session: Any, ctx: Dict[str, Any], facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything the conversation needs to know about who's across the desk."""
    entity = ctx.get("entity") or {}
    pers = dict(ctx.get("personality") or {})
    state = dict(ctx.get("state") or {})
    contract = dict(ctx.get("contract") or {})
    rel = dict(ctx.get("relationship") or {})
    facts = dict(facts or {})
    pid = str(ctx.get("player_id") or "")
    name = str(entity.get("player_name") or "Player")
    ovr = _f(entity.get("overall"), 75.0)
    age = int(_f(ctx.get("age") or entity.get("age"), 26))
    rank, size = _team_rank(session, pid)
    line_rank = int(ctx.get("line_rank") or 0)
    if ctx.get("on_ahl"):
        tier = "prospect" if age <= 23 else "depth"
    elif ovr >= 88 or (rank == 1 and ovr >= 84):
        tier = "franchise"
    elif ovr >= 83 or (0 < rank <= 5):
        tier = "core"
    elif age <= 21 and ovr < 80:
        tier = "prospect"
    elif (0 < rank <= 14) or (0 < line_rank <= 2):
        tier = "regular"
    else:
        tier = "depth"
    ego = _f(pers.get("ego"), 50)
    volatility = _f(pers.get("volatility"), 50)
    professionalism = _f(pers.get("professionalism"), 55)
    years = _f(contract.get("years_remaining"), 2)
    leverage = 0.0
    if years <= 1 and tier in ("franchise", "core"):
        leverage += 0.6  # he can walk next summer
    if contract.get("has_nmc") or contract.get("has_ntc"):
        leverage += 0.2
    broken = int(rel.get("broken_promises") or 0)
    sens = STATURE_SENSITIVITY[tier]
    sens *= 0.7 + ego / 100.0 * 0.6
    sens *= 0.8 + volatility / 100.0 * 0.4
    sens *= 1.15 - professionalism / 100.0 * 0.3
    sens *= 1.0 + leverage * 0.25
    sens = round(max(0.45, min(2.4, sens)), 2)
    morale = _f(state.get("morale"), 55)
    trust = _f(state.get("gm_trust"), 55)
    role_sat = _f(state.get("role_satisfaction"), 60)
    grievance = 0.0
    grievance += max(0.0, 50 - morale) * 0.6
    grievance += max(0.0, 52 - trust) * 0.5
    grievance += max(0.0, 50 - role_sat) * 0.35
    grievance += broken * 7
    pts_pct = _team_points_pct(session)
    gm_pressure = 25.0 + {"franchise": 30, "core": 18, "regular": 8, "depth": 2, "prospect": 10}[tier]
    gm_pressure += leverage * 20
    if pts_pct is not None and pts_pct < 0.47:
        gm_pressure += 10
    gm_pressure += min(15.0, _f(state.get("media_stress"), 30) * 0.12)
    gm_pressure += int(ctx.get("trade_rumor_heat") or 0) * 0.1
    tension = 22.0 + grievance + (sens - 1.0) * 14 + volatility * 0.08
    return {
        "player_id": pid,
        "name": name,
        "first": _first(name),
        "last": _last(name),
        "tier": tier,
        "tier_label": STATURE_LABELS[tier],
        "team_rank": rank,
        "ovr": round(ovr, 1),
        "age": age,
        "ego": ego,
        "volatility": volatility,
        "professionalism": professionalism,
        "accountability": _f(pers.get("accountability"), 55),
        "coachability": _f(pers.get("coachability"), 55),
        "competitiveness": _f(pers.get("competitiveness"), 55),
        "ambition": _f(pers.get("ambition"), 50),
        "loyalty": _f(pers.get("loyalty"), 55),
        "leadership": _f(pers.get("leadership"), 50),
        "money_focus": _f(pers.get("money_focus"), 45),
        "morale": morale,
        "trust": trust,
        "role_sat": role_sat,
        "broken_promises": broken,
        "years_left": years,
        "aav": _f(contract.get("aav_m"), 0.0),
        "leverage": round(leverage, 2),
        "sensitivity": sens,
        "team_pts_pct": pts_pct,
        "facts": facts,
        "line_rank": line_rank,
        "scratched": bool(ctx.get("scratched")),
        "trade_heat": int(ctx.get("trade_rumor_heat") or 0),
        "media_stress": _f(state.get("media_stress"), 30),
        "injured": bool(ctx.get("injured")),
        "gm_pressure": round(_clip(gm_pressure, 5, 95), 1),
        "tension": round(_clip(tension, 8, 88), 1),
    }


def mood_label(tension: float) -> str:
    t = float(tension)
    if t < 25:
        return "Calm"
    if t < 45:
        return "Uneasy"
    if t < 65:
        return "Tense"
    if t < 85:
        return "Heated"
    return "On the edge"


def _stakes_line(mc: Dict[str, Any]) -> str:
    bits: List[str] = []
    tier = mc["tier"]
    if tier == "franchise":
        bits.append(f"{mc['first']} is the face of this team. The room takes its cue from how this goes.")
    elif tier == "core":
        bits.append(f"{mc['first']} is part of the core. Get this wrong and it travels.")
    elif tier == "depth":
        bits.append(f"{mc['first']} is a depth piece. You have leverage, but the room watches how you treat the bottom of the roster.")
    elif tier == "prospect":
        bits.append(f"{mc['first']} is young. What he hears today shapes how he develops.")
    else:
        bits.append(f"{mc['first']} is a regular who wants to know where he fits.")
    if mc["years_left"] <= 1 and tier in ("franchise", "core"):
        bits.append("He's in the last year of his deal, and his agent will hear about this meeting.")
    if mc["broken_promises"]:
        bits.append(f"You've broken {mc['broken_promises']} promise{'s' if mc['broken_promises'] != 1 else ''} to him before. He remembers.")
    if mc["ego"] >= 70 and tier in ("franchise", "core"):
        bits.append("Big ego. Tread carefully.")
    elif mc["volatility"] >= 72:
        bits.append("He runs hot. A wrong word can blow this up.")
    elif mc["professionalism"] >= 72:
        bits.append("A pro. He'll respect a straight answer.")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

_OPEN_NARRATION = {
    "calm": ("He sits down and leans back.", "He drops into the chair and nods at you.", "He's relaxed — this feels like a check-in to him."),
    "uneasy": ("He takes the chair across from you but doesn't settle.", "He shuts the door behind him a little harder than he needs to.", "He's turning his ball cap over in his hands."),
    "tense": ("He doesn't sit down.", "He's standing by the window, arms folded.", "Jaw set. He's been rehearsing this."),
    "hot": ("He comes in already talking.", "He tosses his phone on your desk, screen down.", "He doesn't wait for you to start."),
}

_REACTION_NARRATION = {
    "opens_up": ("His shoulders drop.", "He exhales and finally sits down.", "A half-smile. That landed."),
    "receptive": ("He nods slowly.", "He takes a second with that.", "He's listening now."),
    "guarded": ("He looks past you at the wall.", "Arms stay folded.", "He doesn't say anything for a beat."),
    "bristles": ("His jaw tightens.", "He leans forward over the desk.", "He laughs — not the good kind."),
    "boils_over": ("He's on his feet.", "His voice carries into the hallway.", "He slams a palm on the desk."),
}

_CLOSE_NARRATION = {
    "opens_up": ("He shakes your hand on the way out.", "He leaves lighter than he came in.", "He stops at the door: 'Thanks. Really.'"),
    "receptive": ("He nods and heads out.", "He taps the door frame on his way out.", "'Alright,' he says, and goes."),
    "guarded": ("He leaves without another word.", "He's halfway down the hall before the door closes.", "A short nod, nothing more."),
    "bristles": ("He leaves the door open behind him.", "He mutters something you don't catch.", "He doesn't look back."),
    "boils_over": ("The door rattles in its frame.", "The equipment staff saw him come out. So did two teammates.", "You hear him in the hallway for a while."),
    "walked_out": ("He's gone before you finish the sentence.", "He walks out mid-sentence. The whole floor heard it.", "Chair pushed back, door open, gone."),
}


# What the GM actually says to open each meeting he called. Written as speech, not as the
# menu title ("Tell player they are part of long-term plans" used to be read out loud).
_GM_OPENERS: Dict[str, Tuple[str, ...]] = {
    "discuss_ice_time": ("I want to talk about your minutes and where you fit right now.", "Let's talk about your ice time. I'd rather you hear my thinking from me."),
    "offer_increased_ice_time": ("I think you've earned a bigger piece of the pie. Let's talk about more minutes.", "There's more ice time on the table for you. I want to walk you through it."),
    "explain_reduced_ice_time": ("Your minutes came down. You deserve to know why.", "I know you've seen the time-on-ice sheet. Let me explain it."),
    "discuss_line_assignment": ("I want to talk about where you fit in the lineup.", "Let's talk about your line and what we need from you there."),
    "discuss_promotion": ("You've been pushing for a bigger role. Let's talk about it.", "I've noticed the push you've been making. Let's talk about moving you up."),
    "discuss_demotion": ("I moved you down the lineup. I want to explain it face to face.", "You saw the new lines. I owe you the reasoning."),
    "offer_pp_opportunity": ("I want to talk about the power play, and whether you're part of it.", "Our power play needs a spark. I've been thinking about you."),
    "explain_pp_removal": ("You came off the power play. Let me tell you why.", "I know losing your power-play spot stung. Let's talk."),
    "offer_pk_opportunity": ("I'm thinking about giving you penalty-kill minutes.", "Coaches like your details. Want a crack at the kill?"),
    "discuss_healthy_scratch": ("Sitting you out wasn't easy. I want you to know what it means.", "Let's talk about the scratch. I don't want you guessing."),
    "discuss_return_lineup": ("Good to see you back on your feet. Let's talk about getting you back in.", "The trainers say you're close. Let's map out your return."),
    "discuss_goalie_workload": ("Let's talk about the crease and how we split the starts.", "I want you clear on the plan in net."),
    "praise_performance": ("I wanted to tell you in person: you've been excellent lately.", "Before anything else, the way you've been playing hasn't gone unnoticed up here."),
    "challenge_performance": ("I'm going to be straight with you. Your game hasn't been where it needs to be.", "We need to talk about your play. It's slipped, and you know it."),
    "ask_struggles": ("Something's off with your game. Talk to me. What's going on?", "You don't look like yourself out there. What's happening?"),
    "discuss_scoring_drought": ("The goals have dried up. I want to hear how you're handling it.", "Let's talk about the drought. I'm not here to pile on."),
    "discuss_defensive_struggles": ("Your own-zone game has been costing us. Let's work through it.", "We've been giving up too much when you're out there. Let's talk about it."),
    "discuss_improvement": ("You're trending the right way. I wanted to talk about keeping it going.", "Whatever you changed, it's working. Let's talk about it."),
    "discuss_consistency": ("Some nights you're great, some nights we can't find you. Let's talk about that.", "I want to talk about consistency, because I know what you're capable of."),
    "discuss_playoff_performance": ("This is the time of year that defines guys. Let's talk about your game.", "Playoffs are here. I want to know where your head's at."),
    "address_social_post": ("We need to talk about what you put online.", "I saw the post. So did everybody else. Sit down."),
    "explain_ovr_rising": ("Your game's taken a real step. I want to talk about why.", "You've been better lately, and it's not luck. Let's talk about it."),
    "explain_ovr_falling": ("Your game's dipped. I want to understand it with you.", "You haven't looked like yourself. Let's figure out why."),
    "discuss_development_plan": ("Let's talk about where your game is headed and how we get you there.", "I want to lay out a development plan with you."),
    "set_development_priority": ("If you could only work on one thing this year, I want it to be the right thing. Let's pick it.", "Let's set your development priority for the next stretch."),
    "discuss_stalled_development": ("I think you've hit a wall in your development. Let's talk about getting past it.", "You haven't taken the step we hoped for yet. I want to talk about that."),
    "discuss_aging_decline": ("I want to have an honest conversation about the next couple of years.", "You've been around long enough to know this talk. Let's have it honestly."),
    "discuss_conditioning": ("You've looked worn down. Let's talk about how you're feeling physically.", "I want to check on your conditioning. You've looked tired."),
    "discuss_injury_recovery": ("How's the recovery going? Be honest with me.", "Let's talk about the rehab and how you're really feeling."),
    "ask_extension_interest": ("Your deal's coming up. I want to know if you see yourself here long term.", "Let's talk about an extension. Is staying something you want?"),
    "long_term_plans": ("I wanted you to hear this from me: you're a big part of what we're building here.", "Let me be clear about something. You're in our long-term plans."),
    "future_uncertain": ("I owe you honesty about your future here, even if it's not easy to hear.", "I'm not going to dress this up. Your future here isn't settled."),
    "discuss_role_next_season": ("Let's talk about next season and what your role looks like.", "I want to talk about where you fit next year."),
    "discuss_upcoming_fa": ("Free agency's coming. I'd rather talk about it now than in July.", "Your contract's up after this season. Let's talk about it openly."),
    "hometown_discount": ("I want to talk about your next contract, and what staying here could look like.", "Let's talk about your next deal. I've got something to ask you."),
    "contract_expectations": ("Let's talk numbers, or at least what fair looks like to you.", "I want to hear your expectations on the next deal."),
    "discuss_retirement": ("I want to talk about how much longer you see yourself playing.", "No pressure here. I just want to hear how you're thinking about the end of your career."),
    "ask_trade_welcome": ("I want an honest answer: would a fresh start somewhere else appeal to you?", "Straight question. If the right trade came along, how would you feel about it?"),
    "tell_being_shopped": ("You're going to hear things, so I want you to hear it from me first. Teams have been calling about you.", "I'm not going to hide it. Your name's come up in trade talks, and I've been listening."),
    "reassure_not_shopped": ("I know your name's out there. I want to set the record straight: we're not shopping you.", "Ignore the noise. You're not on the block."),
    "ask_ntc_waiver": ("I have to ask you about your trade protection. I wouldn't if it didn't matter.", "There's no easy way to say this. I need to talk to you about your clause."),
    "ask_preferred_destinations": ("If a move ever happened, I want to know where you'd want to go.", "Hypothetically, if you were moved, where would you want to land?"),
    "discuss_ahl_assignment": ("I want to talk about sending you to the AHL for a stretch.", "We're going to assign you to the AHL. I want you to know why."),
    "discuss_call_up": ("You've been tearing it up down there. Let's talk about the NHL.", "I've been getting good reports on you. Let's talk about a call-up."),
    "address_trade_rumors": ("Your name's been everywhere. Let's talk about what's real.", "I know the rumours have been loud. Let me tell you where things actually stand."),
    "ask_coaching_relationship": ("How are things with you and the coach? Honestly.", "I'm hearing you and the coach aren't on the same page. Talk to me."),
    "ask_locker_room_concerns": ("How's the room? I want the real answer, not the media one.", "You've got a good read on the room. What should I know?"),
    "discuss_leadership": ("Guys follow you. I want to talk about your role as a leader here.", "I want to talk about leadership, and what I see in you."),
    "ask_mentor_younger": ("I've got a favour to ask. One of our young guys could use someone like you.", "I'd like you to take one of the kids under your wing."),
    "personal_wellbeing_check": ("Forget hockey for a minute. How are you doing away from the rink?", "I just want to check in. How's life outside the rink?"),
    "repair_relationship": ("I think things between us got off track. I'd like to fix that.", "We haven't been on great terms. I want to clear the air."),
    "development_meeting": ("Let's talk about your development and what we need you to become.", "I want to sit down and plan out your next steps."),
}

_GM_TOPIC_OPENERS: Dict[str, Tuple[str, ...]] = {
    "role": ("I want to talk about your role and where you fit.", "Let's talk about how we're using you."),
    "performance": ("I want to talk about your game lately.", "Let's talk about how you've been playing."),
    "contract": ("I want to talk about your future here.", "Let's talk about your contract situation."),
    "trade": ("I want to talk about the trade talk around you.", "Let's talk about what you've been hearing."),
    "room": ("I wanted to check in on a few things.", "I've got a few things I want to go over with you."),
}

# The player asked for this one: the GM lets him speak first.
_GM_REQUEST_OPENERS = {
    "calm": ("Come on in, {first}. You wanted to see me?", "Grab a seat, {first}. What's on your mind?", "Door's open, {first}. What can I do for you?"),
    "tense": ("{first}. Sit down. I heard you wanted to talk.", "Come in, {first}. Close the door. What's going on?", "I got your message, {first}. Let's hear it."),
}


def _gm_open_line(mc: Dict[str, Any], title: str, tension: float, seed: str, *, itype: str = "", topic: str = "", requested: bool = False) -> str:
    first = mc["first"]
    if requested:
        pool = _GM_REQUEST_OPENERS["tense" if tension >= 55 else "calm"]
        return _pick(pool, seed, "gmr").format(first=first)
    if tension >= 55:
        greet = _pick((f"Come in, {first}. Shut the door.", f"{first}. Sit down, please.", f"Close the door, {first}."), seed, "gmo")
    else:
        greet = _pick((f"Thanks for coming by, {first}.", f"Come on in, {first}.", f"Got a minute, {first}?"), seed, "gmo")
    body_pool = _GM_OPENERS.get(itype)
    if not body_pool and itype.startswith("habit_"):
        body_pool = ("I want to talk about a change in how you play.", "I've got an adjustment I want you to make to your game.")
    if not body_pool and itype.startswith("rolepact_"):
        body_pool = ("I want to talk about your role, and what I need from you in it.", "Let's get on the same page about the job I'm asking you to do.")
    if not body_pool:
        body_pool = _GM_TOPIC_OPENERS.get(topic) or _GM_TOPIC_OPENERS["room"]
    return f"{greet} {_pick(body_pool, seed, 'gmb', itype)}"


# What the GM says out loud when he picks an answer. The button keeps its short label
# ("Direct accountability"); the transcript gets a sentence a GM would actually say.
_GM_SAY: Dict[str, Tuple[str, ...]] = {
    "discuss_ice_time:honest": ("Honestly? Right now you're where you are because of what I see on the ice. Here's what would change it.",),
    "discuss_ice_time:promise": ("Give me a few weeks. I'll revisit your minutes, and I mean that.",),
    "discuss_ice_time:firm": ("The structure stays as it is for now. Your job is to make it hard on us.",),
    "offer_increased_ice_time:commit": ("You're getting more minutes. You've earned them.",),
    "offer_increased_ice_time:conditional": ("The minutes are there if you keep playing like this. Keep it up and they're yours.",),
    "explain_reduced_ice_time:transparent": ("The coaches wanted a different look, and I agreed. That's the honest reason.",),
    "explain_reduced_ice_time:performance": ("It comes down to how you've played lately. Get your game back and the minutes come back.",),
    "explain_reduced_ice_time:deflect": ("That's really a conversation for the coaching staff.",),
    "discuss_line_assignment:top_six": ("I see a top-six path for you. Here's what it takes to get there.",),
    "discuss_line_assignment:depth": ("Right now you're a depth guy for us, and that's a real job. I need you to own it.",),
    "discuss_promotion:support": ("You're right. You've earned a look higher up, and you're going to get it.",),
    "discuss_promotion:wait": ("Not yet. Keep doing exactly what you're doing and you'll force our hand.",),
    "discuss_demotion:accountability": ("The drop is about your play. Fix the details and you'll move back up.",),
    "discuss_demotion:support": ("It's a short-term adjustment, not a verdict on you. Hang in there.",),
    "offer_pp_opportunity:promise_pp": ("You're getting a real look on the power play. Make it count.",),
    "offer_pp_opportunity:deny": ("Power-play spots get earned at five-on-five first. Show me there.",),
    "explain_pp_removal:explain": ("The unit wasn't producing and we needed a different look. That's it.",),
    "explain_pp_removal:challenge": ("Go win the spot back. I'll notice.",),
    "offer_pk_opportunity:yes": ("I want you on the penalty kill. The coaches trust your details.",),
    "offer_pk_opportunity:no": ("For now I need your energy at even strength.",),
    "discuss_healthy_scratch:development": ("Think of it as a reset, not a punishment. You'll be back in soon.",),
    "discuss_healthy_scratch:performance": ("I'll be honest, it was about performance. You know what needs to improve.",),
    "discuss_return_lineup:welcome": ("Welcome back. You're going right back into your spot.",),
    "discuss_return_lineup:earn": ("The guys filled in well while you were out. You'll have to win your spot back.",),
    "discuss_goalie_workload:promise_starts": ("You're getting a run of starts. The net's yours to grab.",),
    "discuss_goalie_workload:platoon": ("We're splitting the net for now. Here's how we'll decide who gets the big games.",),
    "praise_performance:specific": ("Your work away from the puck has been outstanding. That's what's winning us games.",),
    "praise_performance:team": ("What I love is how you're playing for the guys around you. That rubs off.",),
    "challenge_performance:direct": ("You're better than this, and right now it isn't good enough. I need more.",),
    "challenge_performance:support": ("I expect more from you. And I'll give you whatever you need to get there.",),
    "ask_struggles:listen": ("Take your time. I'm here to listen, not to judge.",),
    "ask_struggles:fix_role": ("Let's change your role a bit and see if it frees you up.",),
    "discuss_scoring_drought:stay_confident": ("Keep shooting. The chances are there, and they'll start going in.",),
    "discuss_scoring_drought:usage_change": ("Let's tweak how we use you. A new look might shake something loose.",),
    "discuss_defensive_struggles:film": ("Let's sit down with video this week and clean up the details.",),
    "discuss_defensive_struggles:pairing": ("I'll talk to the coaches about giving you a partner who complements you.",),
    "discuss_improvement:encourage": ("Keep doing exactly what you're doing. It's working.",),
    "discuss_improvement:raise_bar": ("Good isn't the ceiling for you. I want to see you push for great.",),
    "discuss_consistency:routine": ("Let's look at your routine. Same prep every night, same habits.",),
    "discuss_consistency:pressure": ("I know it's a lot. You don't have to carry all of it by yourself.",),
    "discuss_playoff_performance:belief": ("This is your time. I believe in you in big moments.",),
    "discuss_playoff_performance:prep": ("Let's make sure your prep is airtight. Nothing left to chance.",),
    "address_social_post:take_it_down": ("Take it down today and put out an apology. That's not a request.",),
    "address_social_post:back_him": ("I've got your back on this. Publicly, too.",),
    "address_social_post:posting_ban": ("You're off social media for three weeks. Phone stays in the bag.",),
    "address_social_post:laugh_off": ("It's the internet. Let it blow over and get back to hockey.",),
    "explain_ovr_rising:credit": ("This is on you. You put the work in and it's showing.",),
    "explain_ovr_rising:demand_more": ("Don't let up now. I want to see how far you can take this.",),
    "explain_ovr_falling:support": ("Slumps happen. We'll get you through this one together.",),
    "explain_ovr_falling:accountability": ("Part of this is preparation. I need you showing up ready every day.",),
    "discuss_development_plan:clear_plan": ("Here's the plan: these are the areas we work on, and here's when we check in.",),
    "discuss_development_plan:skills": ("I'm setting you up with focused skills work with the development staff.",),
    "set_development_priority:skating": ("Skating. Get a step quicker and everything else opens up.",),
    "set_development_priority:hockey_sense": ("Your reads. Play faster in your head and the rest will follow.",),
    "discuss_stalled_development:patience": ("Development isn't a straight line. Be patient, it'll come.",),
    "discuss_stalled_development:opportunity": ("I'll find you a real opportunity to break through. Be ready.",),
    "discuss_aging_decline:veteran_role": ("I see a big role for you as a veteran voice. The kids need that.",),
    "discuss_aging_decline:honest": ("I'll be straight: your minutes might come down. I wanted you to hear it from me.",),
    "discuss_conditioning:staff": ("I'm getting you time with the training staff. Let's get your legs back.",),
    "discuss_conditioning:expect": ("I need you ready every night. Here's what that looks like.",),
    "discuss_injury_recovery:no_rush": ("No rush. Your health comes first, full stop.",),
    "discuss_injury_recovery:timeline": ("Let's map out a realistic timeline so you know what's ahead.",),
    "ask_extension_interest:interested": ("We'd like to keep you here. I think there's a deal to be made.",),
    "ask_extension_interest:noncommittal": ("We're keeping our options open for now. Nothing's off the table.",),
    "long_term_plans:commit": ("You're part of the foundation here. I want you for the long haul.",),
    "long_term_plans:cautious": ("I like where this is going. Keep it up and the long term takes care of itself.",),
    "future_uncertain:honest": ("With the cap and our depth chart, I can't promise you a spot next year.",),
    "future_uncertain:soften": ("Whatever happens, you've been a pro here, and you'll be treated with respect.",),
    "discuss_role_next_season:top_role": ("Next year I see you in a top role. Plan for it.",),
    "discuss_role_next_season:depth": ("Next year I see you in a depth role. I'd rather be honest now.",),
    "discuss_upcoming_fa:priority": ("Re-signing you is a priority for me. Let's get something done.",),
    "discuss_upcoming_fa:open": ("Let's keep talking and see where things land.",),
    "hometown_discount:ask": ("I want you here. To make it work under the cap, I need you to take a bit less.",),
    "hometown_discount:respect": ("You've earned your market value. I won't ask you to give that up.",),
    "contract_expectations:listen": ("Tell me what fair looks like to you. I'm listening.",),
    "contract_expectations:anchor": ("I'll be honest: our number is going to be lower than you'd like.",),
    "discuss_retirement:support": ("Whatever you decide, we're behind you.",),
    "discuss_retirement:win_now": ("Give me one more push. Let's go chase it together.",),
    "ask_trade_welcome:explore": ("If you'd be open to a move, let's quietly see what's out there.",),
    "ask_trade_welcome:stay": ("For the record, I'd like you to stay.",),
    "tell_being_shopped:honest": ("It's true. I've been taking calls on you. I didn't want you to hear it elsewhere.",),
    "tell_being_shopped:context": ("It's the business side of the game. It doesn't change how we see you as a person.",),
    "reassure_not_shopped:firm": ("I'm not shopping you. Period. You're staying.",),
    "reassure_not_shopped:vague": ("Nothing's imminent. Don't worry about it.",),
    "ask_ntc_waiver:destination": ("I'm asking you to open up your list. Here's why it would be good for you too.",),
    "ask_ntc_waiver:withdraw": ("Forget I asked. Your clause stands.",),
    "ask_preferred_destinations:listen": ("Tell me where you'd want to go. It stays in this room.",),
    "ask_preferred_destinations:discourage": ("Let's not go down that road. Focus on playing here.",),
    "discuss_ahl_assignment:development": ("This is about getting you big minutes and confidence. It's not the end of anything.",),
    "discuss_ahl_assignment:performance": ("Your play hasn't been NHL level lately. Go earn your way back.",),
    "discuss_call_up:soon": ("Keep this up and you'll get the call soon. I mean that.",),
    "discuss_call_up:work": ("You're close, but not yet. Keep working down there.",),
    "address_trade_rumors:deny": ("There are no active talks. Tune out the noise.",),
    "address_trade_rumors:acknowledge": ("I won't lie to you: in this league anything can happen. But you'll hear it from me first.",),
    "ask_coaching_relationship:mediate": ("Let me get you and the coach in a room together and sort it out.",),
    "ask_coaching_relationship:support_coach": ("I back my coaches. I need you to find a way to make it work with him.",),
    "ask_locker_room_concerns:listen": ("Anything you say stays between us.",),
    "ask_locker_room_concerns:address": ("I'll take this to the leadership group and we'll deal with it.",),
    "discuss_leadership:praise_leader": ("The way you lead in that room matters more than people know.",),
    "discuss_leadership:expect_more": ("I need to hear your voice more in that room. The guys follow you.",),
    "ask_mentor_younger:assign": ("I'd like you to take one of the young guys under your wing.",),
    "ask_mentor_younger:decline": ("On second thought, focus on your own game for now.",),
    "personal_wellbeing_check:support": ("Whatever you need, the team has resources. Use them.",),
    "personal_wellbeing_check:boundaries": ("You don't have to tell me anything. Just know the door's open.",),
    "repair_relationship:apologize": ("I owe you an apology. I haven't handled things well with you.",),
    "repair_relationship:reset": ("Let's reset. Clear expectations, both ways, starting today.",),
    "repair_relationship:deflect": ("I think you're making this bigger than it is.",),
    # Requests he brought to you
    "request_more_ice:honest": ("Here's the honest answer: this is your role right now, and this is what would change it.",),
    "request_more_ice:promise": ("You'll get a bigger opportunity soon. I'll make sure of it.",),
    "request_more_ice:decline": ("Roles here are earned. Keep going and you'll get there.",),
    "request_pp_time:honest": ("Here's how the power-play units shake out right now, and why.",),
    "request_pp_time:promise": ("You'll get a look on the power play. Make it count.",),
    "request_pp_time:decline": ("The units are staying as they are for now.",),
    "request_starting_role:honest": ("Here's the plan in net, and where you fit in it.",),
    "request_starting_role:promise": ("You'll get more starts. Be ready.",),
    "request_starting_role:decline": ("We're sticking with our starter for now.",),
    "contract_clarity:honest": ("I'll be direct about where things stand with your contract.",),
    "contract_clarity:commit": ("You're part of the plan here. I can't talk numbers today, but you're part of it.",),
    "contract_clarity:deflect": ("I'm not ready to commit to anything yet.",),
    "winning_concern_meeting:honest": ("Here's the plan, step by step, and where you fit in it.",),
    "winning_concern_meeting:promise_compete": ("I'll improve this roster. You have my word.",),
    "winning_concern_meeting:dismiss": ("Leave the roster to me. Focus on your game.",),
    "development_meeting:honest": ("Here's exactly what we need from you, and how we'll get you there.",),
    "role_frustration:honest_role": ("I'll be honest about your role, even if it's not what you want to hear.",),
    "role_frustration:promise_opportunity": ("You'll get a bigger opportunity within the next few games.",),
    "role_frustration:challenge_player": ("You want more? Take it. Make it impossible for us to keep you down.",),
    "personal_check_in:support_day": ("Take a day. Family comes first.",),
    "personal_check_in:support_resources": ("We've got people who can help. Let's get you connected.",),
    "personal_check_in:hockey_first": ("I need you to leave it at the door when you come to the rink.",),
    "unheralded_leader:recognize_publicly": ("I'm going to make sure people know what you do for this team.",),
    "unheralded_leader:recognize_privately": ("I see what you do in that room. I wanted you to hear it from me.",),
    # Extra options cut for him
    "ctx_what_he_means": ("You're a huge part of this team, {first}. The room takes its cue from you.",),
    "ctx_earn_it": ("I'll be straight with you: earn it, or you'll be watching from the press box.",),
    "ctx_development_path": ("Here's your path: what to work on, who you'll learn from, and when we'll check in.",),
    "ctx_contract_results": ("You're paid like a top guy. Right now the production isn't matching it.",),
    "ctx_lead_it": ("I want you to lead this in the room. It's your job now.",),
}

_GM_SAY_BY_APPROACH: Dict[str, Tuple[str, ...]] = {
    "support": ("I've got your back on this, {first}.", "I believe in you. That's not changing.", "You've done a lot for this team. I see it."),
    "honest": ("I'll give it to you straight, {first}.", "Here's the honest answer, even if it's not what you want to hear.", "I won't sugarcoat it."),
    "firm": ("That's where it stands, {first}. It's on you now.", "I need more from you. Simple as that.", "This isn't up for debate."),
    "deflect": ("Let's not get into that today.", "That's a conversation for another day.", "I'd rather leave that alone for now."),
    "promise": ("You have my word on this, {first}.", "I'll make it happen. Hold me to it.", "Give me a little time. It's coming."),
}


def _gm_say(mc: Dict[str, Any], choice: Dict[str, Any], approach: str, seed: str) -> str:
    """Spoken line for a GM choice: the template's own line if we wrote one, else a natural
    line for the approach. Never the menu label."""
    cid = str(choice.get("id") or "")
    itype = str(mc.get("itype") or "")
    pool = _GM_SAY.get(f"{itype}:{cid}") or _GM_SAY.get(cid)
    if not pool:
        pool = _GM_SAY_BY_APPROACH.get(approach) or _GM_SAY_BY_APPROACH["honest"]
    return _pick(pool, seed, "say", cid).format(first=mc.get("first") or "")


_SELF_VERBS = (("is", "I'm"), ("has been", "I've been"), ("has", "I have"), ("was", "I was"), ("wants", "I want"), ("needs", "I need"),
               ("knows", "I know"), ("feels", "I feel"), ("thinks", "I think"))


def own_voice(text: str, mc: Dict[str, Any]) -> str:
    """A player never talks about himself by name. Rewrite 'Tyler Kleven's game' -> 'my game'."""
    import re as _re

    out = str(text or "")
    name = str(mc.get("name") or "").strip()
    if not out or not name:
        return out
    forms = [name]
    for part in (mc.get("last"), mc.get("first")):
        if part and len(str(part)) >= 3 and str(part) not in forms:
            forms.append(str(part))
    for form in forms:
        f = _re.escape(form)
        pp = r"(been|got|had|played|done|seen|heard|given|made|earned|never)"
        out = _re.sub(rf"\b{f}['’]s {pp}\b", lambda m: "I've " + m.group(1), out)
        out = _re.sub(rf"(^|[.!?]\s+){f}['’]s\b", lambda m: m.group(1) + "My", out)
        out = _re.sub(rf"\b{f}['’]s\b", "my", out)
        for verb, repl in _SELF_VERBS:
            out = _re.sub(rf"(^|[.!?]\s+){f} {verb}\b", lambda m, r=repl: m.group(1) + r, out)
            out = _re.sub(rf"\b{f} {verb}\b", repl, out)
        out = _re.sub(rf"(^|[.!?]\s+){f}\b", lambda m: m.group(1) + "I", out)
        out = _re.sub(rf"\b{f}\b", "me", out)
    return out


def _pick(pool: Tuple[str, ...], *seed: Any) -> str:
    return pool[_rng(*seed).randrange(len(pool))]


def _fact_hook(mc: Dict[str, Any], topic: str) -> str:
    f = mc.get("facts") or {}
    if not f.get("gp"):
        return ""
    if f.get("goalie"):
        return f"I'm {f.get('line')}."
    if topic == "role":
        if mc["scratched"]:
            return f"I've been watching from the press box. {f.get('pts', 0)} points in {f.get('gp')} games and I'm sitting."
        return f"{f.get('toi_txt')} a night. That's what I'm getting."
    if topic == "performance":
        return f"{f.get('g', 0)} goals, {f.get('pts', 0)} points in {f.get('gp')} games. I know what the numbers say."
    if topic in ("contract", "trade"):
        return f"{f.get('pts', 0)} points in {f.get('gp')} games for this club this year."
    return ""


def _voice_prep(mc: Dict[str, Any], ctx: Dict[str, Any], meeting: Dict[str, Any]) -> None:
    try:
        from app.sim_engine.franchise import meeting_voice as _mv  # noqa: WPS433

        mc.update(_mv.voice_extras(ctx, meeting))
        mc["voice"] = _mv.voice_of(mc)
        mc["voice_label"] = _mv.VOICE_LABELS.get(mc["voice"], "")
    except Exception:
        pass


def _opening_line(mc: Dict[str, Any], base_line: str, topic: str, seed: str) -> str:
    if mc.get("voice"):
        try:
            from app.sim_engine.franchise import meeting_voice as _mv  # noqa: WPS433

            line = _mv.opening_line(mc, topic, seed, base_line)
            if line:
                return line
        except Exception:
            pass
    # The template line may already quote his numbers; don't say them twice.
    hook = "" if any(ch.isdigit() for ch in str(base_line)) else _fact_hook(mc, topic)
    tier, tension = mc["tier"], mc["tension"]
    lead = ""
    if tension >= 65:
        lead = _pick(("Let's not waste each other's time.", "I'm going to say this once.", "I've been sitting on this for a while."), seed, "lead")
    elif tension >= 45:
        lead = _pick(("I appreciate the time, but I need a real answer.", "Okay. Let's do this.", "I've been thinking about this a lot."), seed, "lead")
    elif tension < 25:
        lead = _pick(("Good to see you.", "Thanks for grabbing me.", "What's up?"), seed, "lead")
    star_bit = ""
    if tier == "franchise" and tension >= 45:
        star_bit = _pick(("I've carried a lot of this team.", "You know what I bring here.", "I didn't sign here to be managed like a fourth-liner."), seed, "star")
    elif tier == "depth" and tension >= 45:
        star_bit = _pick(("I know where I sit on the depth chart.", "I'm not asking for the world.", "I just want a fair shake."), seed, "depth")
    return " ".join(b for b in (lead, base_line, hook, star_bit) if b)


# ---------------------------------------------------------------------------
# Choices
# ---------------------------------------------------------------------------

_APPROACH_WORDS = (
    ("deflect", ("deflect", "defer", "avoid", "coaching staff", "later", "dodge", "not now")),
    ("firm", ("firm", "hold", "challenge", "demand", "accountab", "conditional", "earn", "not yet", "wait", "structure", "standard", "bench", "scratch", "send", "cut")),
    ("support", ("praise", "support", "commit", "reassure", "back", "agree", "encourage", "credit", "celebrate", "promote", "trust", "thank")),
    ("honest", ("honest", "transparent", "explain", "assess", "truth", "depth role", "level", "straight", "realistic")),
)


def classify_approach(choice: Dict[str, Any]) -> str:
    outcome = dict(choice.get("outcome") or {})
    if outcome.get("promise"):
        return "promise"
    text = f"{choice.get('id') or ''} {choice.get('label') or ''} {choice.get('detail') or ''}".lower()
    for approach, words in _APPROACH_WORDS:
        if any(w in text for w in words):
            return approach
    total = 0.0
    for bucket in (outcome.get("profile_changes") or {}).values():
        if isinstance(bucket, dict):
            total += sum(_f(v, 0) for k, v in bucket.items() if "morale" in str(k) or "trust" in str(k))
    if total <= -6:
        return "firm"
    if total >= 4:
        return "support"
    return "honest"


def _fit(mc: Dict[str, Any], approach: str) -> float:
    """How well this approach lands with this player, about -1..1 before noise."""
    ego, vol, pro = mc["ego"] / 100.0, mc["volatility"] / 100.0, mc["professionalism"] / 100.0
    acc, coach, comp = mc["accountability"] / 100.0, mc["coachability"] / 100.0, mc["competitiveness"] / 100.0
    tier = mc["tier"]
    score = {"support": 0.25, "honest": 0.15, "firm": -0.05, "deflect": -0.35, "promise": 0.3}.get(approach, 0.0)
    if approach == "support":
        score += ego * 0.35 - (0.2 if tier == "depth" and acc > 0.65 else 0.0)
    elif approach == "honest":
        score += pro * 0.45 + acc * 0.25 - ego * 0.25
    elif approach == "firm":
        score += comp * 0.25 + acc * 0.3 + coach * 0.3 - ego * 0.55 - vol * 0.3
        score += {"depth": 0.3, "prospect": 0.15, "regular": 0.05, "core": -0.15, "franchise": -0.4}[tier]
    elif approach == "deflect":
        score -= max(0.0, 55 - mc["trust"]) / 100.0
    elif approach == "promise":
        score += mc["ambition"] / 100.0 * 0.2 - mc["broken_promises"] * 0.3
    if mc["morale"] < 40:
        score -= 0.1
    if mc["trust"] >= 70:
        score += 0.15
    elif mc["trust"] < 40:
        score -= 0.15
    return score


def _read_for(mc: Dict[str, Any], approach: str) -> Tuple[str, str]:
    """A hint the GM would have from knowing the player — never the numbers."""
    fit = _fit(mc, approach)
    risky = mc["sensitivity"] >= 1.3
    if fit >= 0.45:
        return "He'll appreciate it", "safe"
    if fit >= 0.15:
        return "Should land", "safe"
    if fit >= -0.15:
        return ("Hard to read him" if not risky else "Could go either way"), "uncertain"
    if fit >= -0.45:
        return ("Risky with him" if risky else "He won't love it"), "risky"
    return ("He might blow up" if risky else "This will sting"), "danger"


def _extra_choices(mc: Dict[str, Any], topic: str, base: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One extra, situation-specific option on top of the template's answers."""
    first = mc["first"]
    out: List[Dict[str, Any]] = []
    tier = mc["tier"]
    if tier in ("franchise", "core"):
        out.append({
            "id": "ctx_what_he_means",
            "label": f"Tell {first} what he means to this team",
            "detail": "Make it about him and the room, not the problem.",
            "outcome": {"profile_changes": {"actor": {"state.morale": 3, "state.belonging": 3, "state.gm_trust": 2}}, "relationship": {"loyalty": 3}},
            "approach": "support",
        })
    elif tier == "depth":
        out.append({
            "id": "ctx_earn_it",
            "label": "Earn it or watch from the press box",
            "detail": "No cushioning. He knows where he stands.",
            "outcome": {"profile_changes": {"actor": {"state.focus": 4, "state.morale": -3, "state.gm_trust": -1}}, "relationship": {"respect": 2}},
            "approach": "firm",
        })
    elif tier == "prospect":
        out.append({
            "id": "ctx_development_path",
            "label": f"Lay out {first}'s development path",
            "detail": "Specifics: what to work on, who he learns from, when he's re-evaluated.",
            "outcome": {"profile_changes": {"actor": {"state.focus": 4, "state.confidence": 2, "state.gm_trust": 2}}, "relationship": {"communication": 3}},
            "approach": "honest",
        })
    if mc["aav"] >= 6.0 and topic == "performance" and (mc.get("facts") or {}).get("ppg", 1.0) < 0.55:
        out.append({
            "id": "ctx_contract_results",
            "label": "Point to the contract and the results",
            "detail": f"${mc['aav']:.1f}M a year. The production isn't there and you say so.",
            "outcome": {"profile_changes": {"actor": {"state.focus": 3, "state.morale": -4, "state.gm_trust": -3}}, "relationship": {"respect": 1}},
            "approach": "firm",
        })
    if mc["age"] >= 31 and mc["leadership"] >= 60 and topic in ("room", "role"):
        out.append({
            "id": "ctx_lead_it",
            "label": "Ask him to lead it in the room",
            "detail": "Turn the complaint into a job.",
            "outcome": {"profile_changes": {"actor": {"state.belonging": 4, "state.gm_trust": 2}}, "team_changes": {"unity": 2}, "relationship": {"respect": 2}},
            "approach": "honest",
        })
    taken = {str(c.get("id")) for c in base}
    return [c for c in out if c["id"] not in taken][:1]


def _cast_label(mc: Dict[str, Any], choice: Dict[str, Any], approach: str) -> str:
    """Re-cut the button text for who's in the chair."""
    label = str(choice.get("label") or "")
    tier, first = mc["tier"], mc["first"]
    if approach == "firm" and tier == "franchise":
        return f"{label} — even for {first}"
    if approach == "firm" and tier == "depth":
        return f"{label}. No cushioning"
    if approach == "support" and tier == "depth":
        return f"{label} (he'll notice you made the time)"
    if approach == "deflect" and tier in ("franchise", "core"):
        return f"{label} (he'll see through it)"
    return label


def tailor_choices(mc: Dict[str, Any], choices: List[Dict[str, Any]], topic: str, seed: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in list(choices or []) + _extra_choices(mc, topic, list(choices or [])):
        c2 = copy.deepcopy(c)
        approach = str(c2.get("approach") or classify_approach(c2))
        read, risk = _read_for(mc, approach)
        c2["approach"] = approach
        if not str(c2.get("id") or "").startswith("ctx_"):
            c2["label"] = _cast_label(mc, c2, approach)
        c2["read"] = read
        c2["risk"] = risk
        c2["tone"] = approach
        if not c2.get("say"):
            c2["say"] = _gm_say(mc, c2, approach, seed)
        c2.pop("effect_preview", None)
        c2.pop("effect_summary", None)
        out.append(c2)
    return out


# ---------------------------------------------------------------------------
# Outcome scaling
# ---------------------------------------------------------------------------


def scale_outcome(node: Any, pos_mult: float, neg_mult: float, key: str = "") -> Any:
    """Scale good effects by pos_mult and bad effects by neg_mult (respecting keys where lower is better)."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k == "promise" or k in _SKIP_SCALE:
                out[k] = copy.deepcopy(v)
            else:
                out[k] = scale_outcome(v, pos_mult, neg_mult, str(k))
        return out
    if isinstance(node, list):
        return [scale_outcome(x, pos_mult, neg_mult, key) for x in node]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return node
    good = (node < 0) if key in _LOWER_IS_BETTER else (node > 0)
    return round(float(node) * (pos_mult if good else neg_mult), 2)


def _add(outcome: Dict[str, Any], bucket: str, field: str, delta: float, sub: str = "actor") -> None:
    if bucket == "profile_changes":
        pc = outcome.setdefault("profile_changes", {})
        actor = pc.setdefault(sub, {})
        actor[field] = round(_f(actor.get(field), 0) + delta, 2)
    else:
        b = outcome.setdefault(bucket, {})
        b[field] = round(_f(b.get(field), 0) + delta, 2)


# ---------------------------------------------------------------------------
# Beats
# ---------------------------------------------------------------------------


def dress_opening(session: Any, ctx: Dict[str, Any], meeting: Dict[str, Any], topic: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Beat 0: who's in the chair, how tense it is, his opening, and options cut for him."""
    mc = build_context(session, ctx, facts)
    _voice_prep(mc, ctx, meeting)
    seed = f"{meeting.get('id')}"
    tension = mc["tension"]
    bucket = "calm" if tension < 25 else "uneasy" if tension < 45 else "tense" if tension < 65 else "hot"
    name = mc["name"]
    dialogue = list(meeting.get("dialogue") or [])
    # Replace the player's canned opener with a contextual one, after a stage direction.
    base_line = ""
    for line in dialogue:
        if isinstance(line, dict) and str(line.get("speaker") or "") == name:
            base_line = str(line.get("text") or "")
            break
    others = [
        ln for ln in dialogue
        if isinstance(ln, dict) and str(ln.get("speaker") or "") not in ("GM", "You", name, "narration")
    ]
    narration = {"speaker": "narration", "text": _pick(_OPEN_NARRATION[bucket], seed, "open")}
    if others:
        # Multi-party scene (two teammates, a reporter): keep every voice, set the mood first.
        meeting["dialogue"] = [narration] + [ln for ln in dialogue if isinstance(ln, dict)]
    else:
        opener = own_voice(_opening_line(mc, base_line, topic, seed), mc)
        gm_line = next((ln for ln in dialogue if isinstance(ln, dict) and str(ln.get("speaker") or "") == "GM"), None)
        requested = str(meeting.get("initiator") or "") != "gm"
        new_dialogue: List[Dict[str, Any]] = []
        if gm_line or requested:
            new_dialogue.append({"speaker": "GM", "text": _gm_open_line(
                mc, str(meeting.get("title") or ""), tension, seed,
                itype=str(mc.get("itype") or meeting.get("interaction_type") or meeting.get("kind") or ""),
                topic=topic, requested=requested,
            )})
        new_dialogue.append(narration)
        new_dialogue.append({"speaker": name, "text": opener, "mood": mood_label(tension)})
        meeting["dialogue"] = new_dialogue
    meeting["choices"] = tailor_choices(mc, list(meeting.get("choices") or []), topic, seed)
    meeting["conversation"] = {
        "tension": tension,
        "tension_start": tension,
        "gm_pressure": mc["gm_pressure"],
        "mood": mood_label(tension),
        "stature": mc["tier"],
        "stature_label": mc["tier_label"],
        "stakes": _stakes_line(mc),
        "voice": mc.get("voice"),
        "voice_label": mc.get("voice_label"),
        "sensitivity": mc["sensitivity"],
        "topic": topic,
        "beats": 0,
        "history": [{"beat": 0, "tension": tension}],
    }
    return meeting


def _roll_reaction(mc: Dict[str, Any], approach: str, seed: str) -> str:
    fit = _fit(mc, approach) + _rng(seed, "react").uniform(-0.18, 0.18)
    if fit >= 0.45:
        return "opens_up"
    if fit >= 0.15:
        return "receptive"
    if fit >= -0.15:
        return "guarded"
    if fit >= -0.45:
        return "bristles"
    return "boils_over"


_REACTION_LINES = {
    "role": {
        "opens_up": ("Okay. That's all I wanted — to know it's real.", "Alright. I can work with that. I'll show you.", "That's fair. I'll be ready when the chance comes."),
        "receptive": ("I hear you. I just need to see it on the sheet.", "Fine. But I'm going to hold you to it.", "Okay. Next week tells me if you meant it."),
        "guarded": ("We'll see.", "I've heard versions of this before.", "Sure. Talk is easy, though."),
        "bristles": ("So that's it? I just eat the minutes?", "You'd say that to anyone in this chair.", "Respectfully, that's not an answer."),
        "boils_over": ("Then trade me. I'm serious.", "Don't sell me a plan you don't have.", "You want me to smile while I rot on the fourth line?"),
    },
    "performance": {
        "opens_up": ("Yeah. I needed to hear that.", "I know. I've been grinding on it — thanks for seeing it.", "Okay. Let's fix it."),
        "receptive": ("Fair. I'll own that.", "I know I've got more.", "Alright, I'll take it."),
        "guarded": ("Everybody's an expert this year.", "I know what I'm doing out there.", "Okay."),
        "bristles": ("You're putting this on me? Look at who I'm playing with.", "I don't need a lecture.", "That's a convenient read."),
        "boils_over": ("You've got some nerve.", "Say that in front of the room, then.", "Maybe you should look at your own decisions."),
    },
    "contract": {
        "opens_up": ("That's what I wanted to hear. My agent will be glad too.", "Okay. I want to be here — let's make it work.", "Good. I can relax a bit now."),
        "receptive": ("Alright. Let's keep talking.", "Fair. Get my agent on the phone.", "Okay. Let's see the numbers."),
        "guarded": ("I'll pass that along.", "We'll see what the market says.", "Noted."),
        "bristles": ("That's not how you treat a guy who's given you what I have.", "Then I guess I'll test the market.", "My agent won't love that."),
        "boils_over": ("Then we're done talking. Call my agent.", "You'll regret that number.", "I'm not taking a discount to be disrespected."),
    },
    "trade": {
        "opens_up": ("That means a lot. I want to stay.", "Okay. I'll stop reading the rumours.", "Good. I'm in."),
        "receptive": ("Alright. Just don't blindside me.", "Fair enough. Keep me in the loop.", "Okay."),
        "guarded": ("Everyone says that until the call comes.", "We'll see.", "Sure."),
        "bristles": ("So I should start packing?", "That's not reassuring.", "Thanks for nothing."),
        "boils_over": ("Then just do it. Move me.", "I'm done finding this out from Twitter.", "Unbelievable."),
    },
    "room": {
        "opens_up": ("Yeah. I can do that.", "Thanks for asking me instead of telling me.", "Alright. I've got it."),
        "receptive": ("Okay. I'll handle my end.", "Fair.", "I hear you."),
        "guarded": ("Sure.", "If you say so.", "We'll see how it goes."),
        "bristles": ("Why is this on me?", "That's not my problem to fix.", "Come on."),
        "boils_over": ("You don't know what goes on in that room.", "Don't drag me into your mess.", "I'm not your messenger."),
    },
}


def _context_jab(mc: Dict[str, Any], topic: str, reaction: str, seed: str, approach: str = "") -> str:
    """What else is on his mind, pulled from his actual situation."""
    hot = reaction in ("bristles", "boils_over")
    cold = reaction == "guarded"
    warm = reaction in ("opens_up", "receptive")
    options: List[str] = []
    if mc["broken_promises"] and (hot or cold):
        options.append("You told me something like this before, and it didn't happen.")
    if mc["trade_heat"] >= 35 and (hot or cold):
        options.append("And I keep reading my name in trade talk. Nobody here's said a word to me about it.")
    if mc["years_left"] <= 1 and mc["tier"] in ("franchise", "core") and (hot or cold):
        options.append("My agent's going to ask me how this went. What do I tell him?")
    pct = mc.get("team_pts_pct")
    if pct is not None and pct < 0.45 and hot and mc["tier"] in ("franchise", "core", "regular"):
        options.append("We're losing, and this is where you spend your energy?")
    if mc["scratched"] and hot:
        options.append("I'm sitting in the press box in a suit. You get that, right?")
    if mc["media_stress"] >= 65 and (hot or cold):
        options.append("The media's been on me all week. This doesn't help.")
    if mc["injured"] and topic in ("role", "performance") and not warm:
        options.append("I'm playing hurt, for what it's worth.")
    if warm and mc["trust"] >= 65:
        options.append("You've always been straight with me. I appreciate that.")
    if warm and mc["tier"] == "depth" and approach in ("support", "honest", "promise"):
        options.append("Not every GM makes time for the guys at the bottom of the lineup.")
    if warm and mc["age"] <= 22:
        options.append("I just want to get better. Tell me what to work on and I'll do it.")
    if not options:
        return ""
    # Not every line needs a second thought; about two in three do.
    r = _rng(seed, "jab")
    if r.random() > 0.66:
        return ""
    return options[r.randrange(len(options))]


def _reaction_line(mc: Dict[str, Any], topic: str, reaction: str, seed: str, approach: str = "") -> str:
    pool = (_REACTION_LINES.get(topic) or _REACTION_LINES["room"]).get(reaction) or ("Okay.",)
    line = _pick(pool, seed, "rl")
    if mc.get("voice") and (_rng(seed, "voice_mix").random() < 0.7 or str(mc.get("itype") or "").startswith("habit_")):
        try:
            from app.sim_engine.franchise import meeting_voice as _mv  # noqa: WPS433

            line = _mv.reaction_line(mc, topic, reaction, seed, approach) or line
        except Exception:
            pass
    if "fourth line" in line and mc["tier"] not in ("depth",) and int(mc.get("line_rank") or 0) not in (3, 4):
        line = line.replace("rot on the fourth line", "watch my minutes disappear")
    jab = _context_jab(mc, topic, reaction, seed, approach)
    return f"{line} {jab}".strip()


def _followups(mc: Dict[str, Any], base_choice: Dict[str, Any], reaction: str, topic: str, tension: float) -> List[Dict[str, Any]]:
    """Beat 2 options, shaped by how he took beat 1. Each one gives up something (M4)."""
    base = copy.deepcopy(base_choice.get("outcome") or {})
    pm, nm, _ = REACTION_EFFECT[reaction]
    base = scale_outcome(base, pm, nm)
    cid = str(base_choice.get("id") or "c")
    first, tier, sens = mc["first"], mc["tier"], mc["sensitivity"]

    def mk(sid: str, label: str, detail: str, outcome: Dict[str, Any], tension_delta: float, read: str, risk: str) -> Dict[str, Any]:
        # "Hold the line: 'That's where it is.'" -> the button keeps it, the GM says the quote.
        say = label
        if ": '" in label and label.endswith("'"):
            say = label.split(": '", 1)[1][:-1]
        elif label.startswith("'") and label.endswith("'"):
            say = label[1:-1]
        if sid == "give":
            say = "Look, I hear you. Let me see what I can do on my end."
        return {
            "id": f"{cid}__{sid}",
            "label": label,
            "say": say,
            "detail": detail,
            "outcome": outcome,
            "tension_delta": round(tension_delta, 1),
            "read": read,
            "risk": risk,
            "tone": sid,
        }

    if reaction in ("opens_up", "receptive"):
        lock = copy.deepcopy(base)
        push = scale_outcome(copy.deepcopy(base), 1.15, 1.2)
        _add(push, "profile_changes", "state.focus", 3)
        _add(push, "relationship", "respect", 2)
        leave = scale_outcome(copy.deepcopy(base), 0.9, 0.9)
        _add(leave, "relationship", "trust", 2)
        lines = {
            "role": ("That's the deal. Go earn it.", "Then I want more from you on the backcheck too.", "Good. That's all for today."),
            "performance": ("Next game. Show me.", "And I want you leading the effort, not just the scoring.", "Good talk. Go get some rest."),
            "contract": ("Then let's get it done.", "And I need you to be patient on structure.", "Good. Leave the rest to me and your agent."),
            "trade": ("You're part of this. Act like it.", "Then help me sell the room on what we're building.", "That's all I wanted to say."),
            "room": ("That's the job.", "And I'll be checking in on it.", "Good. Thanks for coming in."),
        }.get(topic) or ("That's the deal.", "And I need more from you.", "Good. That's it for today.")
        return [
            mk("lock", lines[0], "Seal it as is.", lock, -6, "Safe", "safe"),
            mk("push", lines[1], "Ask more of him while he's open to it.", push, 6 * sens, "He might feel squeezed" if sens >= 1.3 else "He can take it", "uncertain" if sens >= 1.3 else "safe"),
            mk("leave", lines[2], "Don't oversell it.", leave, -10, "Safe", "safe"),
        ]
    if reaction == "guarded":
        level = scale_outcome(copy.deepcopy(base), 1.0, 0.8)
        _add(level, "relationship", "trust", 3)
        hold = scale_outcome(copy.deepcopy(base), 1.0, 1.2)
        _add(hold, "relationship", "respect", 2)
        _add(hold, "relationship", "trust", -2)
        give = scale_outcome(copy.deepcopy(base), 1.2, 1.0)
        _add(give, "profile_changes", "state.morale", 2)
        _add(give, "relationship", "respect", -2)
        return [
            mk("level", f"Level with him: 'I'm not playing games, {first}.'", "Drop the script and be straight.", level, -12, "Should land", "safe"),
            mk("hold", "Hold the line: 'That's where it is.'", "No movement. He can take it or leave it.", hold, 10 * sens, "Risky with him" if sens >= 1.3 else "He'll accept it", "risky" if sens >= 1.3 else "uncertain"),
            mk("give", "Give him something to walk out with", "A small concession. He'll take it, and remember you blinked.", give, -8, "He'll take it", "safe"),
        ]
    # bristles / boils_over
    cool = scale_outcome(copy.deepcopy(base), 0.6, 0.6)
    _add(cool, "relationship", "respect", -2)
    ground = scale_outcome(copy.deepcopy(base), 1.0, 1.3)
    _add(ground, "relationship", "respect", 3 if tier in ("depth", "prospect", "regular") else -2)
    options = [
        mk("cool", "Lower the temperature: 'Sit down. Let's start over.'", "He calms down. He also saw you back off.", cool, -22, "Calms it down", "safe"),
        mk("ground", "Hold your ground: 'Watch your tone. I'm not done.'", "You don't flinch. With him, that could go either way.", ground, 18 * sens, "He might walk" if tension + 18 * sens >= 95 else "Risky", "danger" if tension + 18 * sens >= 95 else "risky"),
    ]
    if tier in ("franchise", "core"):
        flatter = scale_outcome(copy.deepcopy(base), 1.1 if mc["ego"] >= 60 else 0.8, 0.8)
        _add(flatter, "profile_changes", "state.morale", 2)
        _add(flatter, "relationship", "loyalty", 2)
        options.append(mk("flatter", f"'{first}, you're the guy here. That's why I'm in this room with you.'", "Appeal to his place on the team.", flatter, -14 if mc["ego"] >= 60 else -4, "He likes hearing it" if mc["ego"] >= 60 else "He may see through it", "safe" if mc["ego"] >= 60 else "uncertain"))
    else:
        bluff = scale_outcome(copy.deepcopy(base), 0.9, 1.1)
        _add(bluff, "relationship", "respect", 3)
        _add(bluff, "profile_changes", "state.morale", -2)
        options.append(mk("bluff", "Call his bluff: 'The door's right there.'", "You have the leverage. Use it.", bluff, 8, "He'll back down" if mc["volatility"] < 65 else "Could blow up", "uncertain" if mc["volatility"] < 65 else "risky"))
    return options


def react(session: Any, ctx: Dict[str, Any], meeting: Dict[str, Any], choice: Dict[str, Any], topic: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Beat 1: how he takes it, the tension shift, his line, and the follow-ups."""
    mc = build_context(session, ctx, facts)
    _voice_prep(mc, ctx, meeting)
    conv = dict(meeting.get("conversation") or {})
    tension = _f(conv.get("tension"), mc["tension"])
    approach = str(choice.get("approach") or classify_approach(choice))
    seed = f"{meeting.get('id')}|{choice.get('id')}"
    reaction = _roll_reaction(mc, approach, seed)
    delta = REACTION_EFFECT[reaction][2]
    if delta > 0:
        delta *= mc["sensitivity"]
    tension = round(_clip(tension + delta, 0, 100), 1)
    name = mc["name"]
    dialogue = list(meeting.get("dialogue") or [])
    dialogue.append({"speaker": "GM", "text": str(choice.get("say") or _gm_say(mc, choice, approach, seed))})
    dialogue.append({"speaker": "narration", "text": _pick(_REACTION_NARRATION[reaction], seed, "rn")})
    dialogue.append({"speaker": name, "text": own_voice(_reaction_line(mc, topic, reaction, seed, approach), mc), "mood": mood_label(tension), "reaction": reaction})
    conv.update({
        "tension": tension,
        "mood": mood_label(tension),
        "reaction": reaction,
        "reaction_label": REACTION_LABELS[reaction],
        "approach": approach,
        "beats": 1,
    })
    conv.setdefault("history", []).append({"beat": 1, "tension": tension, "reaction": reaction})
    meeting["dialogue"] = dialogue
    meeting["conversation"] = conv
    meeting["choices"] = _followups(mc, choice, reaction, topic, tension)
    meeting["beat"] = 1
    meeting["status"] = "pending"
    return meeting


_WALKOUT_LINES = (
    "We're done here.",
    "I don't have to sit here for this.",
    "Talk to my agent.",
    "You want to play it that way? Fine.",
)


def finalize(session: Any, ctx: Dict[str, Any], meeting: Dict[str, Any], choice: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Beat 2: final tension, walk-out check, and the closing scene. Returns (choice_to_apply, scene)."""
    mc = build_context(session, ctx, {})
    conv = dict(meeting.get("conversation") or {})
    tension = _clip(_f(conv.get("tension"), mc["tension"]) + _f(choice.get("tension_delta"), 0), 0, 100)
    reaction = str(conv.get("reaction") or "receptive")
    seed = f"{meeting.get('id')}|{choice.get('id')}|close"
    walked = tension >= 100
    final = dict(choice)
    outcome = copy.deepcopy(choice.get("outcome") or {})
    if walked:
        outcome = scale_outcome(outcome, 0.0, 1.25)
        _add(outcome, "profile_changes", "state.morale", -3)
        _add(outcome, "profile_changes", "state.gm_trust", -4)
        _add(outcome, "relationship", "grievance", 5)
        outcome.pop("promise", None)
        outcome["public"] = bool(mc["volatility"] >= 60 or mc["tier"] in ("franchise", "core"))
        reaction = "walked_out"
    else:
        # The last answer can still shift the read: easing tension lifts it, spiking it sours it.
        td = _f(choice.get("tension_delta"), 0)
        idx = REACTION_ORDER.index(reaction) if reaction in REACTION_ORDER else 2
        if td <= -12 and idx < len(REACTION_ORDER) - 1:
            idx += 1
        elif td >= 15 and idx > 0:
            idx -= 1
        reaction = REACTION_ORDER[idx]
    final["outcome"] = outcome
    name = mc["name"]
    if walked:
        closing_line = _pick(_WALKOUT_LINES, seed, "w")
    else:
        closing_line = {
            "opens_up": ("Thanks. I mean it.", "Okay. We're good.", "Appreciate you."),
            "receptive": ("Alright. We'll see how it goes.", "Fair enough.", "Okay. Talk soon."),
            "guarded": ("Fine.", "Whatever you say.", "Got it."),
            "bristles": ("Yeah. Okay.", "Sure. We'll see.", "Message received."),
            "boils_over": ("This isn't over.", "Don't expect me to forget this.", "Unbelievable."),
        }[reaction]
        closing_line = _pick(closing_line, seed, "cl")
    verdict = {
        "opens_up": f"{mc['first']} left believing you.",
        "receptive": f"{mc['first']} is on board — for now.",
        "guarded": f"{mc['first']} didn't buy all of it.",
        "bristles": f"{mc['first']} left frustrated.",
        "boils_over": f"{mc['first']} left angry.",
        "walked_out": f"{mc['first']} walked out on you.",
    }[reaction]
    said = str(choice.get("say") or _gm_say(mc, choice, str(choice.get("approach") or classify_approach(choice)), seed))
    closing_line = own_voice(closing_line, mc)
    scene = {
        "said": said,
        "quote": closing_line,
        "closing": [
            {"speaker": "GM", "text": said},
            {"speaker": name, "text": closing_line, "mood": mood_label(tension)},
            {"speaker": "narration", "text": _pick(_CLOSE_NARRATION[reaction], seed, "cn")},
        ],
        "verdict": verdict,
        "reaction": reaction,
        "reaction_label": REACTION_LABELS[reaction],
        "tension_end": round(tension, 1),
        "tension_start": conv.get("tension_start"),
        "walked_out": walked,
        "stature": mc["tier"],
    }
    return final, scene
