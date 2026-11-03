"""Natural, personality-driven meeting dialogue (template based, no LLM).

``meeting_conversation`` builds a context dict (``mc``) for whoever is across the desk.
This module turns that into human lines: a *voice* (hothead, pro, star, kid, veteran,
leader, quiet, joker) picked from his personality and age, the *ask* for the specific
meeting type, and one optional *context clause* drawn from his real situation (stat
line, ice time, contract year, where the team sits). Every pick is seeded by the
meeting id so a conversation reads the same on reload but differs between meetings.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["voice_of", "opening_line", "reaction_line", "voice_extras", "VOICE_LABELS"]


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:12], 16))


def _pick(pool: Sequence[str], *seed: Any) -> str:
    pool = [p for p in pool if p]
    if not pool:
        return ""
    return pool[_rng(*seed).randrange(len(pool))]


def _f(v: Any, d: float = 50.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


VOICE_LABELS = {
    "hothead": "Runs hot",
    "star": "Star ego",
    "pro": "Consummate pro",
    "kid": "Young and eager",
    "veteran": "Seen-it-all veteran",
    "leader": "Room leader",
    "quiet": "Quiet type",
    "joker": "Loose in the room",
    "grinder": "Lunch-pail guy",
}


def voice_of(mc: Dict[str, Any]) -> str:
    """One dominant voice from personality, age and stature."""
    vol = _f(mc.get("volatility"))
    ego = _f(mc.get("ego"))
    pro = _f(mc.get("professionalism"), 55)
    lead = _f(mc.get("leadership"))
    age = int(_f(mc.get("age"), 26))
    tier = str(mc.get("tier") or "regular")
    if vol >= 70:
        return "hothead"
    if ego >= 70 and tier in ("franchise", "core"):
        return "star"
    if age <= 22:
        return "kid"
    if age >= 32:
        return "veteran"
    if lead >= 70:
        return "leader"
    if pro >= 70:
        return "pro"
    if tier == "depth" and _f(mc.get("competitiveness"), 55) >= 55:
        return "grinder"
    if ego <= 38 and lead <= 45:
        return "quiet"
    if _f(mc.get("media_stress"), 30) <= 25 and vol <= 45 and ego >= 45:
        return "joker"
    return "pro" if pro >= 55 else "grinder"


def voice_extras(ctx: Dict[str, Any], meeting: Dict[str, Any]) -> Dict[str, Any]:
    """Situation facts the base context dict doesn't carry."""
    contract = dict(ctx.get("contract") or {})
    pst = dict(ctx.get("pst") or {})
    return {
        "itype": str(meeting.get("interaction_type") or meeting.get("kind") or ""),
        "on_pp": bool(ctx.get("on_pp")),
        "pk_unit": int(ctx.get("pk_unit") or 0),
        "on_ahl": bool(ctx.get("on_ahl")),
        "is_goalie": bool(ctx.get("is_goalie")),
        "improving": bool(ctx.get("improving")),
        "struggling": bool(ctx.get("struggling")),
        "drought": bool(ctx.get("scoring_drought")),
        "has_extension": bool(contract.get("has_extension")),
        "clause": str(contract.get("clause_label") or ""),
        "walkouts": int(pst.get("meeting_walkouts") or 0),
        "position": str(ctx.get("position") or ""),
        "directive_misfit": _directive_misfit(ctx, meeting),
    }


def _directive_misfit(ctx: Dict[str, Any], meeting: Dict[str, Any]) -> float:
    itype = str(meeting.get("interaction_type") or "")
    if not itype.startswith("habit_"):
        return 0.0
    try:
        from app.sim_engine.franchise.player_directives import misfit  # noqa: WPS433

        return float(misfit(itype, ctx.get("player")))
    except Exception:
        return 0.3


# ---------------------------------------------------------------------------
# Openers by voice x tension bucket
# ---------------------------------------------------------------------------

_LEADS: Dict[str, Dict[str, Sequence[str]]] = {
    "hothead": {
        "calm": ("Yeah, I've got a minute.", "What's up, boss?", "Sure. Let's hear it."),
        "uneasy": ("Alright, let's get into it.", "I figured you'd call me in eventually.", "Okay. I've got some things to say too."),
        "tense": ("I'm going to be blunt, because that's who I am.", "I've been biting my tongue for weeks.", "You want honesty? Here it is."),
        "hot": ("I'm not going to sit here and nod.", "Don't give me the speech.", "I'm already past polite on this one."),
    },
    "star": {
        "calm": ("Good to see you. What do you need from me?", "Happy to talk — I know this team runs through my line.", "Sure, let's chat."),
        "uneasy": ("I'll be honest, I expected this conversation sooner.", "Let's make this quick, I've got treatment after.", "I've been patient. I'd like that noticed."),
        "tense": ("I don't think you realize how this looks from where I sit.", "I came here to win, not to be managed.", "I've earned the right to be straight with you."),
        "hot": ("I'm not some kid you can shuffle around.", "My agent already knows I'm in here, just so you know.", "Respectfully — I carry this team, and I'm treated like I don't."),
    },
    "pro": {
        "calm": ("Thanks for making the time.", "Appreciate you grabbing me.", "Always good to check in."),
        "uneasy": ("I'd like to understand the thinking, that's all.", "I'll keep this professional, but I do have concerns.", "I want to be clear about where I stand."),
        "tense": ("I'll say this respectfully, but I need a real answer.", "I don't like bringing things upstairs. That should tell you something.", "I've thought about how to say this for a while."),
        "hot": ("I've tried to handle this the right way. It isn't working.", "I'm done being diplomatic about it.", "With respect, this has gone on too long."),
    },
    "kid": {
        "calm": ("Hey — thanks for having me up here.", "This is kind of cool, honestly. What's up?", "Hi. Am I in trouble?"),
        "uneasy": ("I, uh — I wanted to ask you something.", "Sorry, I'm not great at these.", "My dad said I should just ask, so I'm asking."),
        "tense": ("I don't want to sound ungrateful, but...", "I know I'm young. I'm not stupid, though.", "Can I be honest without it hurting me?"),
        "hot": ("I feel like nobody tells me anything.", "I'm trying to do everything right and it doesn't matter.", "I'm sorry, I'm just really frustrated."),
    },
    "veteran": {
        "calm": ("I've been around long enough to know a check-in when I see one.", "Pull up a chair, I'm not going anywhere.", "Good. I like talking face to face."),
        "uneasy": ("I've seen how these go. Let's skip to the real part.", "I'm too old to dance around things.", "I've sat in this chair in three different buildings."),
        "tense": ("I've given this league a lot of years. I'd like a straight answer.", "Don't manage me. Talk to me.", "I've earned the truth, whatever it is."),
        "hot": ("I didn't play this long to be treated like a spare part.", "Twelve years in, and this is how it goes?", "I'm not going to pretend this is fine."),
    },
    "leader": {
        "calm": ("Glad we're doing this. The room's talking, so it's good timing.", "Hey. Happy to help however I can.", "Thanks. I'd rather hear things from you directly."),
        "uneasy": ("I'm here for me, but also for the guys.", "I want to talk about this before it spreads.", "There's a feeling in the room you should know about."),
        "tense": ("I've been holding the room together. I need something back.", "I can only tell the guys 'stay patient' so many times.", "If I'm frustrated, imagine the younger guys."),
        "hot": ("I'm the one answering questions in the room, not you.", "I can't sell this to the group anymore.", "I've defended this front office. I'm not sure I can keep doing it."),
    },
    "quiet": {
        "calm": ("Hey.", "Sure. What's up?", "Okay."),
        "uneasy": ("I don't usually come up here.", "I'm not big on talking, so... I'll just say it.", "This is a little awkward for me."),
        "tense": ("I've been quiet about it. That doesn't mean it's fine.", "I don't make noise. Maybe I should have.", "I'll keep it short."),
        "hot": ("I'm usually the last guy to complain. I'm complaining.", "I've had enough.", "I'm not going to say this twice."),
    },
    "joker": {
        "calm": ("Is this where you tell me I'm getting a raise?", "Nice office. Smaller than I pictured.", "If this is about the shaving cream in the coach's skates, it wasn't me."),
        "uneasy": ("Okay, no jokes. What's going on?", "You've got the serious face on. Alright.", "I'll behave. Mostly."),
        "tense": ("I'll be serious for once — this bugs me.", "Not much funny about this one.", "I laugh a lot. I'm not laughing about this."),
        "hot": ("Joke's over. I'm actually angry.", "You know it's bad when I'm the one yelling.", "Nobody's laughing in the room right now, including me."),
    },
    "grinder": {
        "calm": ("Thanks for the time. I know you're busy.", "Hey. Just tell me what you need.", "Sure thing."),
        "uneasy": ("I just want to know where I stand.", "I'm not asking for a lot here.", "I work hard. I think everybody sees that."),
        "tense": ("I block shots, I kill penalties, I do the dirty work. I need something.", "I've done everything asked. Everything.", "I'm not a star, I know that. I still deserve a straight answer."),
        "hot": ("I bleed for this team and I get this?", "I'm out there taking hits for guys who get the credit.", "I'm done being the easy guy to push around."),
    },
}


def _bucket(tension: float) -> str:
    t = float(tension)
    return "calm" if t < 25 else "uneasy" if t < 45 else "tense" if t < 65 else "hot"


# ---------------------------------------------------------------------------
# The ask: what this meeting is about, in his words
# ---------------------------------------------------------------------------

_ASKS: Dict[str, Sequence[str]] = {
    # role & usage
    "discuss_ice_time": ("I want to know where I really stand with the minutes.", "Am I getting a fair look out there, or am I just filling a spot?", "Tell me honestly how you see my role right now."),
    "offer_increased_ice_time": ("If there are more minutes on the table, I want them. What does that actually look like?", "You're saying more ice — I'd like to hear specifics.", "More minutes? I'm listening. Tell me how it works."),
    "explain_reduced_ice_time": ("My minutes went down and nobody told me why.", "I looked at the sheet. That's a big drop. Why?", "I can handle bad news. I can't handle no news. Why'd my ice go down?"),
    "discuss_line_assignment": ("Where do you actually see me in this lineup?", "I'd like to know which line you think I belong on.", "Am I a top-six guy to you or not?"),
    "discuss_promotion": ("I think I've outgrown the line I'm on.", "I'm ready for more. I think the tape shows it.", "Move me up. I'll make it work."),
    "discuss_demotion": ("Getting bumped down a line stings. I'd like to understand it.", "I saw the new lines. Not going to lie, that hurt.", "Why'd I get dropped?"),
    "offer_pp_opportunity": ("Give me a real look on the power play and I'll produce.", "I can help the power play. I just need the chance.", "I see the half-wall open on the PP every night. Let me take it."),
    "explain_pp_removal": ("I lost my spot on the power play. What happened?", "I'd like to know why I'm off the PP.", "One bad stretch and I'm off the power play?"),
    "offer_pk_opportunity": ("I can help on the kill if you trust me with it.", "Put me on the PK. I'll block anything.", "I want penalty-kill minutes. I think I can be good at it."),
    "discuss_healthy_scratch": ("Sitting in the press box in a suit is killing me. What's the message?", "I got scratched and I don't know what I'm supposed to fix.", "What do I have to do to get back in the lineup?"),
    "discuss_return_lineup": ("I'm ready to come back. What's the plan for me?", "Trainers cleared me. Where do I slot in?", "I'm healthy. Let's talk about getting me back out there."),
    "discuss_goalie_workload": ("I want clarity on the crease.", "Who's the guy? I'd like to know.", "Rotations are fine, but I want to know where I stand."),
    "request_more_ice": ("I want more responsibility. If I haven't earned it, tell me exactly what I'm missing.", "I think I'm ready for a bigger role, and I want a straight answer on it.", "Give me more minutes and I'll show you what I can do with them."),
    "request_pp_time": ("Give me a real look on the power play. I think I can help us there.", "Our power play's stuck. I think I can unstick it.", "I'd like a shot on the first unit. Even a few games."),
    "request_starting_role": ("I want the net more often. I think I've earned the chance to run with it.", "I'm ready to be the starter. Give me a run of games.", "I play better when I play a lot. Let me prove it."),
    # performance
    "praise_performance": ("Appreciate you noticing — that means something.", "Thanks. It's been fun lately, honestly.", "Glad it's showing. I've been putting the work in."),
    "challenge_performance": ("I know I haven't been good enough.", "Yeah, I've seen the tape. It's not great.", "I know why I'm here. I've been bad."),
    "ask_struggles": ("Honestly? A few things are off.", "It's a bunch of little things. I'm squeezing the stick.", "I don't totally know. Confidence, maybe."),
    "discuss_scoring_drought": ("The points aren't coming. Believe me, I know.", "I'm getting chances. They're just not going in.", "I'm snakebit. I hit three posts last week."),
    "discuss_defensive_struggles": ("My defensive game hasn't been sharp.", "I've been chasing the play too much.", "I know I've been on for too many goals against."),
    "discuss_improvement": ("Feels like things are finally clicking.", "I feel faster out there. Like the game slowed down.", "It's coming together. I can feel it."),
    "discuss_consistency": ("I know I need to be more consistent.", "Two good games, one bad. I hate it too.", "I'm streaky. I know that about myself."),
    "discuss_playoff_performance": ("Playoffs are a different animal.", "This is what you play for. I want to be at my best.", "Every shift matters now. I feel it."),
    # development
    "explain_ovr_rising": ("I've felt the difference lately.", "I feel like a different player this year.", "Something clicked this summer."),
    "explain_ovr_falling": ("I know something's off.", "I don't feel like myself out there.", "My legs aren't there. Or my head isn't."),
    "discuss_development_plan": ("Tell me what the organization needs me to become.", "What's the plan for me? Like, actually?", "I want a real roadmap, not just 'keep working'."),
    "set_development_priority": ("Which part of my game should I be hammering in the summer?", "If I'm working on one thing, what should it be?", "Tell me the one skill that gets me to the next level."),
    "discuss_stalled_development": ("I feel like I've plateaued.", "I'm not getting better as fast as I want to.", "I'm stuck. I don't know how else to say it."),
    "discuss_aging_decline": ("I know the clock is ticking.", "I'm not the same player I was at 27. I'm not blind.", "Let's talk about what the next couple years look like for me."),
    "discuss_conditioning": ("I'll do whatever it takes to be ready.", "My body's been beat up. I'm working on it.", "I'm tired. I'll be honest about that."),
    "discuss_injury_recovery": ("I'm doing everything I can to get back right.", "Rehab's slow. It's driving me nuts.", "I hate watching. I want back in."),
    "development_meeting": ("Tell me what the organization needs me to become. I want a real plan.", "I want to know what I'm working toward here.", "Where's my development headed? I'd like to hear it from you."),
    # contract
    "ask_extension_interest": ("I'm open to talking about staying — depending on the terms.", "I like it here. I'd like to keep liking it.", "My agent's been asking. What are you thinking?"),
    "long_term_plans": ("That's what I needed to hear.", "Good. I want to be part of what's being built here.", "That means a lot, actually."),
    "future_uncertain": ("That's hard to hear.", "So I should be preparing for anything.", "Okay. At least I know."),
    "discuss_role_next_season": ("What role do you see for me next year?", "Where do I fit next season?", "Am I in the plan next year, and doing what?"),
    "discuss_upcoming_fa": ("Free agency's on my mind. I won't pretend it isn't.", "This is my last year. I'm thinking about July.", "My deal's up at the end of the year and my agent's calling every week."),
    "hometown_discount": ("I've given a lot to this organization.", "I love it here. That's not the same as working for free.", "I know where this is going."),
    "contract_expectations": ("Let's talk about what fair looks like.", "I want to be paid like what I am to this team.", "I'm not trying to break the bank. I just want fair."),
    "discuss_retirement": ("I'm thinking about how much longer I want to do this.", "My kids are getting older. It's on my mind.", "I want to go out the right way, whenever that is."),
    "contract_clarity": ("I don't need a number today. I need to know whether I'm part of the plan.", "Am I in your plans after this year? I need to know.", "My agent keeps asking me what you're thinking. I don't have an answer."),
    # trade
    "ask_trade_welcome": ("If it's the right situation, I'm willing to listen.", "I'm not demanding anything. But I'd listen.", "If you think it's better for both of us, say so."),
    "tell_being_shopped": ("So it's true. You're shopping me.", "I heard. Guys talk.", "I'd rather hear it from you than from my agent."),
    "reassure_not_shopped": ("Good. I needed to hear that.", "Okay. Because my name's everywhere right now.", "Thanks. My family's been asking."),
    "ask_ntc_waiver": ("That's a big ask. I'd need a good reason.", "I negotiated that clause for a reason.", "You want me to open up my list? Convince me."),
    "ask_preferred_destinations": ("If a move happens, here's what I'd consider.", "I've thought about it. A few places make sense.", "I'd want somewhere I can win."),
    "discuss_ahl_assignment": ("Sending me down is a message. I want to hear it straight.", "The AHL? Really?", "I'll go. I just want to know why and for how long."),
    "discuss_call_up": ("I'm ready for the NHL.", "I'm tearing it up down there. I'm ready.", "When do I get my shot?"),
    "address_trade_rumors": ("The rumors are everywhere. What's true?", "I can't open my phone without seeing my name.", "Just tell me what's real."),
    "winning_concern_meeting": ("I want to win. I need to understand what we're building toward.", "I didn't come here to miss the playoffs every year.", "What's the plan? Because from the room, it doesn't look like there is one."),
    # team / personal
    "ask_coaching_relationship": ("My relationship with the coach is... complicated.", "Me and the coach aren't seeing eye to eye.", "I don't think the coach trusts me."),
    "ask_locker_room_concerns": ("There are some things in the room worth talking about.", "The room's not great right now.", "Guys are frustrated. I'll leave names out of it."),
    "discuss_leadership": ("Leadership matters on this team.", "I take the letter seriously, or I would if I had it.", "I try to set the tone. I'd like to keep doing that."),
    "ask_mentor_younger": ("I can help the young guys.", "I remember being 19 in this league. I'd like to help.", "Happy to take a kid under my wing."),
    "personal_wellbeing_check": ("I appreciate you asking about off the ice.", "It's been a lot at home. Thanks for asking.", "Honestly, it's been a tough few weeks away from the rink."),
    "personal_check_in": ("Thanks for checking in. It's been a heavy stretch.", "Off-ice stuff has been rough. I'm handling it.", "I'm okay. Mostly. Thanks for asking."),
    "repair_relationship": ("We should talk about where things stand between us.", "I think we got off track somewhere.", "I'd like to clear the air if you're up for it."),
    "address_social_post": ("I'm guessing this is about what I posted.", "Yeah. The post.", "I know why I'm here."),
}

_TOPIC_ASK_FALLBACK = {
    "role": ("I want to talk about my role.", "I'd like to know where I fit."),
    "performance": ("I know my game's been under the microscope.", "Let's talk about how I've been playing."),
    "contract": ("Let's talk about my future here.", "I want to know where things stand contract-wise."),
    "trade": ("I've been hearing things.", "Let's talk about the rumors."),
    "room": ("There's something on my mind.", "I've got a few things to say."),
}

# Voice-flavoured add-ons after the ask (optional, ~half the time).
_TAGS: Dict[str, Sequence[str]] = {
    "hothead": ("And I'm not the only one who thinks so.", "Don't tell me to be patient.", "I'll say it louder in the room if I have to."),
    "star": ("You know what I bring here.", "I'm not asking for a favour.", "Look at who drives the bus around here."),
    "pro": ("Whatever the answer is, I'll respect it.", "I'll take a hard truth over a nice story.", "I just want to know what's expected."),
    "kid": ("I'll do whatever you tell me, I promise.", "I just want to get better.", "Sorry if that's dumb."),
    "veteran": ("I've seen how this goes in other places.", "I'm not going to be around forever, so let's be honest.", "I've been on both sides of this."),
    "leader": ("The guys are watching how this goes.", "It's bigger than me, is the thing.", "I need something I can bring back to the room."),
    "quiet": ("That's it, really.", "Yeah.", ""),
    "joker": ("Also, the coffee machine's broken again. Separate issue.", "I'll go back to being funny after this.", ""),
    "grinder": ("I'll keep showing up either way.", "Just be straight with me.", "I don't need much. Just the truth."),
}


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _context_clause(mc: Dict[str, Any], topic: str, itype: str, seed: str) -> str:
    """One line from his real situation. Picks only facts that are true."""
    f = dict(mc.get("facts") or {})
    gp = int(_f(f.get("gp"), 0))
    opts: List[str] = []
    voice = mc.get("voice") or "pro"
    years = _f(mc.get("years_left"), 2)
    pct = _num(mc.get("team_pts_pct"))
    goalie = bool(f.get("goalie") or mc.get("is_goalie"))
    # Stat line
    if gp >= 5 and not goalie:
        pts, g, toi = int(_f(f.get("pts"), 0)), int(_f(f.get("g"), 0)), str(f.get("toi_txt") or "")
        ppg = pts / max(1, gp)
        if topic in ("role",) and toi:
            if mc.get("scratched"):
                opts.append(f"I've got {pts} points in {gp} games and I'm watching from upstairs.")
            elif ppg >= 0.75:
                opts.append(f"I'm at {pts} points in {gp} games on {toi} a night. Imagine what I'd do with more.")
            else:
                opts.append(f"I'm getting {toi} a night. Hard to make a difference with that.")
        elif topic == "performance" or itype.startswith("habit_"):
            if ppg >= 0.9:
                opts.append(f"{pts} points in {gp} games — I know I'm hot right now.")
            elif ppg < 0.35 and gp >= 10:
                opts.append(f"{g} goals in {gp} games. I know. Trust me, I know.")
            else:
                opts.append(f"{g} goals, {pts} points in {gp} games. It's fine. It's not what I want.")
        elif topic in ("contract", "trade"):
            opts.append(f"I've put up {pts} points in {gp} games for this club this year.")
    elif gp >= 5 and goalie and f.get("line"):
        opts.append(f"I'm {f.get('line')}.")
    # Ice time / deployment
    if topic == "role" and not goalie:
        lr = int(_f(mc.get("line_rank"), 0))
        if lr >= 3 and mc.get("tier") in ("franchise", "core"):
            opts.append("A guy like me on the third line? Come on.")
        if mc.get("on_pp") is False and itype in ("offer_pp_opportunity", "request_pp_time", "explain_pp_removal"):
            opts.append("I watch the power play from the bench and I see what I'd do.")
        if mc.get("on_ahl"):
            opts.append("I'm doing everything right down there.")
    # Contract
    if topic == "contract":
        if years <= 1.05 and not mc.get("has_extension"):
            opts.append("It's my contract year. Everyone keeps reminding me.")
        elif years >= 3:
            opts.append(f"I've still got {int(years)} years left, so this isn't about leaving.")
        aav = _f(mc.get("aav"), 0)
        if aav and aav < 2.0 and int(_f(f.get("pts"), 0)) >= 40:
            opts.append("I'm producing like a top guy and getting paid like a depth guy.")
    elif years <= 1.05 and not mc.get("has_extension") and mc.get("tier") in ("franchise", "core") and voice in ("star", "hothead"):
        opts.append("And with my deal up, I'm paying attention to how I'm treated.")
    # Team standing
    if pct is not None:
        if pct < 0.45 and topic in ("trade", "room", "contract"):
            opts.append(_pick(("We're losing, and losing gets old fast.", "We're near the bottom. That wears on you.", "Nobody likes looking at the standings right now."), seed, "lose"))
        elif pct >= 0.62 and topic in ("role", "room"):
            opts.append(_pick(("We're winning, and I want to be a bigger part of it.", "Things are good here. I don't want to be the guy on the outside of it.", "Team's rolling. I want in on that."), seed, "win"))
    if mc.get("broken_promises"):
        opts.append("Last time I sat here, I was told something that didn't happen.")
    if int(_f(mc.get("trade_heat"), 0)) >= 35 and topic != "trade":
        opts.append("And I keep seeing my name in trade rumours, which doesn't help.")
    if not opts:
        return ""
    r = _rng(seed, "ctx")
    if r.random() > 0.8:
        return ""
    return opts[r.randrange(len(opts))]


def opening_line(mc: Dict[str, Any], topic: str, seed: str, base_line: str = "") -> str:
    """His first line. Voice lead + specific ask + one true fact + voice tag."""
    voice = mc.get("voice") or voice_of(mc)
    mc["voice"] = voice
    itype = str(mc.get("itype") or "")
    bucket = _bucket(_f(mc.get("tension"), 35))
    lead = _pick((_LEADS.get(voice) or _LEADS["pro"])[bucket], seed, "lead", voice)
    ask_pool = _ASKS.get(itype)
    if ask_pool is None and itype.startswith("habit_"):
        try:
            from app.sim_engine.franchise.player_directives import directive_opening_pool  # noqa: WPS433

            ask_pool = directive_opening_pool(mc, itype)
        except Exception:
            ask_pool = None
    if ask_pool is None and itype.startswith("rolepact_"):
        try:
            from app.sim_engine.franchise.role_acceptance import role_opening_pool  # noqa: WPS433

            ask_pool = role_opening_pool(mc, itype)
        except Exception:
            ask_pool = None
    if not ask_pool:
        if base_line and not any(ch.isdigit() for ch in base_line):
            ask_pool = (base_line,)
        else:
            ask_pool = _TOPIC_ASK_FALLBACK.get(topic) or _TOPIC_ASK_FALLBACK["room"]
    ask = _pick(ask_pool, seed, "ask", itype)
    clause = _context_clause(mc, topic, itype, seed)
    tag = ""
    if _rng(seed, "tag").random() < 0.45 and bucket != "calm" or (voice in ("kid", "joker") and _rng(seed, "tag2").random() < 0.35):
        tag = _pick(_TAGS.get(voice) or (), seed, "tagp")
    # Hot players lead with the grievance; calm ones with the greeting.
    parts = [lead, ask, clause, tag] if bucket != "hot" else [ask, lead, clause, tag]
    out = " ".join(p for p in parts if p)
    return out.replace("  ", " ").strip()


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------

_VOICE_REACTIONS: Dict[str, Dict[str, Sequence[str]]] = {
    "hothead": {
        "opens_up": ("...Okay. Yeah. That I can live with.", "Fine. That's actually fair. Don't make me regret it.", "Alright. I came in hot. That helps."),
        "receptive": ("I'll take it. For now.", "Okay. But I'm watching.", "Fine. Show me."),
        "guarded": ("Whatever.", "Yeah, we'll see.", "Heard that one before."),
        "bristles": ("Are you kidding me?", "That's a joke, right?", "You'd never say that to the stars on this team."),
        "boils_over": ("You know what? Forget it.", "Then why am I even here?", "Unbelievable. Unbelievable."),
    },
    "star": {
        "opens_up": ("Good. That's how it should be.", "That's what I expected to hear. Thank you.", "Okay. Now we're on the same page."),
        "receptive": ("Fine. I'll give it some time.", "Alright. I'll hold you to it.", "That's a start."),
        "guarded": ("I'll think about it.", "My agent will want to hear this.", "We'll see if that's real."),
        "bristles": ("I don't think you understand who you're talking to.", "That's not how you treat your best player.", "I've carried this team, and that's the answer?"),
        "boils_over": ("Then maybe I should be somewhere that values me.", "Call my agent. We're done.", "You're going to regret that."),
    },
    "pro": {
        "opens_up": ("Thank you. That's all I wanted — a straight answer.", "I appreciate that. I'll be ready.", "That's fair. I can work with that."),
        "receptive": ("Understood. I'll do my part.", "Okay. I hear you.", "Fair enough."),
        "guarded": ("I'll take that under advisement.", "I understand. I don't agree, but I understand.", "Alright."),
        "bristles": ("With respect, that doesn't really answer it.", "I expected more than that, honestly.", "That's disappointing."),
        "boils_over": ("I've always handled this the right way. I'm not sure that's mattered.", "I think we're done for today.", "I need to step away before I say something I regret."),
    },
    "kid": {
        "opens_up": ("Really? Okay — wow. Thank you.", "That's awesome. I won't let you down.", "Okay! Yeah. I can do that."),
        "receptive": ("Okay. I'll work on it.", "Yeah, that makes sense.", "Okay. Thanks for telling me."),
        "guarded": ("Oh. Okay.", "I guess.", "Sure."),
        "bristles": ("That's... not really fair.", "I thought I was doing well, though.", "I don't get it."),
        "boils_over": ("I'm trying so hard and it doesn't matter!", "Nobody here believes in me.", "Forget it. I'll just go."),
    },
    "veteran": {
        "opens_up": ("Fair. I appreciate the honesty — not everybody gives it.", "That's how you treat a guy. Thanks.", "Alright. I'll help any way I can."),
        "receptive": ("Okay. I've heard worse.", "Fair enough. I'll keep doing my thing.", "Alright. Let's see where it goes."),
        "guarded": ("I've heard that line in three cities.", "Mm-hm.", "Sure. We'll see."),
        "bristles": ("I'm too old to get jerked around.", "I've been doing this longer than you've had this job.", "That's a young GM answer."),
        "boils_over": ("I don't need this. I've got a house in the off-season and a family that misses me.", "Then let me go somewhere that wants me.", "We're done here."),
    },
    "leader": {
        "opens_up": ("Good. I can bring that back to the guys.", "That's going to help the room. Thank you.", "Okay. I'll make sure everyone hears it."),
        "receptive": ("Alright. I'll pass that along.", "Okay. The room will want to see it, though.", "Fair. I'll do my part."),
        "guarded": ("I don't know how I sell that to the guys.", "The room won't love that.", "Okay. I'll think about how to frame it."),
        "bristles": ("I can't take that back to the room.", "You're making my job in there harder.", "The guys are going to hear about this either way."),
        "boils_over": ("Then you go tell them.", "I'm not covering for this anymore.", "I'm done carrying the message."),
    },
    "quiet": {
        "opens_up": ("...Thanks.", "Okay. Good.", "Yeah. That helps."),
        "receptive": ("Okay.", "Alright.", "Got it."),
        "guarded": ("...", "Sure.", "Okay, I guess."),
        "bristles": ("That's not right.", "Hm.", "No."),
        "boils_over": ("I'm leaving.", "We're done.", "Don't."),
    },
    "joker": {
        "opens_up": ("Look at us, communicating. Thanks — really.", "Okay, that was actually nice. Don't tell anyone.", "See, this is why I like you."),
        "receptive": ("Fair. I'll stop complaining. Probably.", "Alright, deal.", "Okay, I can live with that."),
        "guarded": ("Sure, boss.", "Mm. Noted.", "Yeah, okay."),
        "bristles": ("Not funny.", "Wow. Okay.", "That one actually stings."),
        "boils_over": ("I'm not joking anymore.", "You've managed to make the funny guy angry. Congrats.", "Yeah, no. I'm out."),
    },
    "grinder": {
        "opens_up": ("That's all I needed. Thanks.", "Appreciate it. I'll keep working.", "Means a lot coming from you."),
        "receptive": ("Okay. I'll keep doing my job.", "Fair. I'll be ready.", "Alright."),
        "guarded": ("Okay.", "If you say so.", "I'll keep showing up."),
        "bristles": ("I do everything you ask and I get that?", "That's rough.", "Come on, man."),
        "boils_over": ("Find somebody else to block shots, then.", "I'm done being the easy guy.", "That's it. I'm done."),
    },
}

_TOPIC_TAILS: Dict[str, Dict[str, Sequence[str]]] = {
    "role": {
        "warm": ("I'll make the most of the minutes.", "Next game, watch.", "I'll be ready when my number's called."),
        "cold": ("I'll be looking at the TOI sheet.", "Minutes don't lie.", ""),
    },
    "performance": {
        "warm": ("I'll be better. I already know what to fix.", "I'll go watch the tape tonight.", ""),
        "cold": ("I know what I'm doing out there.", "Hockey's a long season.", ""),
    },
    "contract": {
        "warm": ("Let's get something done.", "My agent will be glad to hear it.", "I'd like to stay. Really."),
        "cold": ("We'll see what July says.", "My agent will call you.", ""),
    },
    "trade": {
        "warm": ("I can stop checking my phone, then.", "I want to be here. I'll leave it at that.", ""),
        "cold": ("Don't let me find out on social media.", "I'll pack light, I guess.", ""),
    },
    "room": {
        "warm": ("I'll handle my end.", "I've got it.", ""),
        "cold": ("I'll do my job. That's it.", "", ""),
    },
}


def reaction_line(mc: Dict[str, Any], topic: str, reaction: str, seed: str, approach: str = "") -> str:
    """His answer to the GM's choice in his own voice, with a topic tail."""
    voice = mc.get("voice") or voice_of(mc)
    pools = _VOICE_REACTIONS.get(voice) or _VOICE_REACTIONS["pro"]
    line = _pick(pools.get(reaction) or ("Okay.",), seed, "vr", voice)
    tails = _TOPIC_TAILS.get(topic) or _TOPIC_TAILS["room"]
    warm = reaction in ("opens_up", "receptive")
    tail = ""
    if reaction != "boils_over" and _rng(seed, "tail").random() < 0.55:
        tail = _pick(tails["warm" if warm else "cold"], seed, "tl")
    itype = str(mc.get("itype") or "")
    if itype.startswith("habit_"):
        try:
            from app.sim_engine.franchise.player_directives import directive_reaction_tail  # noqa: WPS433

            tail = directive_reaction_tail(mc, itype, reaction, seed) or tail
        except Exception:
            pass
    elif itype.startswith("rolepact_"):
        try:
            from app.sim_engine.franchise.role_acceptance import role_reaction_tail  # noqa: WPS433

            tail = role_reaction_tail(mc, itype, reaction, seed) or tail
        except Exception:
            pass
    return " ".join(p for p in (line, tail) if p).strip()
