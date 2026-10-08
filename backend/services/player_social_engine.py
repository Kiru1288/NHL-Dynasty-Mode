"""Players on social media.

Every NHL player has an account with a voice that comes from his personality
(``universe_players[pid]["personality"]``) and his current state (morale, coach and
GM trust, media stress). Posts react to what actually happened: his line tonight,
the score, his ice time, his mood, trades and injuries.

What makes it more than a highlight feed:

* **Personas.** Loose cannons, hype men, PR-trained vets, family guys, meme lords,
  cryptic posters, lifestyle accounts, grinders and lurkers all post differently.
* **Chaos.** Volatile, high-ego, low-professionalism players rant after losses,
  call out refs, teammates, the coach or the fans, then delete it — and the
  screenshot accounts are always faster. They guarantee wins (and get the receipts
  posted when they lose), chirp opponents, like posts calling for the coach's job,
  and unfollow the team account when they've had enough.
* **Secret burners.** A few players run anonymous accounts that defend themselves
  in the third person, shade the coach about ice time and leak the room. Fan
  sleuths build cases on IceHole (sometimes accusing the wrong guy) until the
  account gets unmasked.
* **Consequences.** Incidents move the player's media stress, coach/GM trust,
  morale and fan sentiment, and the big ones become storylines. On your club they
  open a "social media" conversation in the meetings room.

Entry points (called from ``social_feed_engine``): ``gen_player_voices`` (played
day), ``gen_player_event_voices`` (current day, events only),
``react_to_gm_burner`` and ``player_social_summary``.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from services import social_feed_engine as sfe

_log = logging.getLogger(__name__)

PS_KEY = "ps"
MAX_INCIDENTS = 40

PERSONA_LABELS = {
    "loose_cannon": "Loose cannon",
    "hype_man": "Hype man",
    "corporate": "PR-trained",
    "family_man": "Family guy",
    "meme_lord": "Meme lord",
    "cryptic": "Cryptic poster",
    "lifestyle": "Lifestyle account",
    "lurker": "Lurker",
    "grinder": "All business",
}

# How often each persona posts on a normal day (before situation multipliers).
PERSONA_ACTIVITY = {
    "loose_cannon": 1.5, "hype_man": 1.4, "corporate": 0.7, "family_man": 0.9, "meme_lord": 1.3,
    "cryptic": 0.6, "lifestyle": 1.1, "lurker": 0.15, "grinder": 0.45,
}

SLEUTH_ACCOUNTS = [
    {"id": "biowatch", "name": "Bio Watch", "handle": "@BioWatchHKY"},
    {"id": "unfollowtracker", "name": "Unfollow Tracker", "handle": "@UnfollowTracker"},
    {"id": "screenshotarchive", "name": "Screenshot Archive", "handle": "@DeletedHockey"},
    {"id": "burnerhunters", "name": "Burner Hunters", "handle": "@BurnerHunters"},
]

BURNER_HANDLES = [
    "@sauce_merchant_77", "@nofilter_hockey", "@icecold_takes_", "@notmymain_acct", "@benchview_9", "@stickflex_truth",
    "@silent_assist", "@rinkrat_unfiltered", "@toi_police", "@slapshot_sage", "@grinder_gospel", "@backcheck_bandit",
    "@thirdperiod_ghost", "@pressbox_pigeon", "@tape_to_tape_tea", "@blueline_burner", "@zamboni_confessions", "@onetimer_oracle",
]

DOG_NAMES = ["Moose", "Biscuit", "Puck", "Tilly", "Bear", "Gordie", "Mabel", "Zamboni", "Ziggy", "Hank"]

INCIDENT_FX: Dict[str, Dict[str, float]] = {
    # state deltas (0-100 scale) and fan sentiment
    "rant": {"media_stress": 7, "focus": -2, "fan": -4},
    "rant_coach": {"media_stress": 9, "coach_trust": -6, "fan": -3},
    "rant_teammates": {"media_stress": 8, "belonging": -6, "fan": -5},
    "rant_fans": {"media_stress": 10, "fan": -12},
    "unfollow": {"media_stress": 8, "belonging": -5, "gm_trust": -3, "fan": -8},
    "like": {"media_stress": 5, "coach_trust": -3, "fan": -2},
    "guarantee_fail": {"media_stress": 6, "confidence": -4, "fan": -6},
    "guarantee_win": {"confidence": 4, "morale": 3, "fan": 6},
    "burner_unmasked": {"media_stress": 18, "morale": -8, "belonging": -9, "coach_trust": -8, "gm_trust": -4, "fan": -15},
    "cryptic": {"media_stress": 3, "fan": -1},
}


# ---------------------------------------------------------------------------
# State and lookups
# ---------------------------------------------------------------------------


def _ps(st: Dict[str, Any]) -> Dict[str, Any]:
    ps = st.get(PS_KEY)
    if not isinstance(ps, dict):
        ps = {"personas": {}, "burners": {}, "guarantees": [], "incidents": [], "cooldown": {}, "season": None}
        st[PS_KEY] = ps
    for k, d in (("personas", {}), ("burners", {}), ("guarantees", []), ("incidents", []), ("cooldown", {})):
        ps.setdefault(k, d)
    return ps


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


def _entities(session: Any) -> Dict[str, Dict[str, Any]]:
    return getattr(session, "universe_players", None) or {}


def _by_team(F: "sfe._Pass") -> Dict[str, List[Dict[str, Any]]]:
    cache = getattr(F, "_ps_by_team", None)
    if cache is None:
        cache = {}
        for e in _entities(F.session).values():
            if not isinstance(e, dict) or not bool(e.get("active_roster", True)):
                continue
            cache.setdefault(str(e.get("team_id") or ""), []).append(e)
        F._ps_by_team = cache
    return cache


def _cooldown_ok(ps: Dict[str, Any], key: str, day: int, days: int) -> bool:
    last = ps["cooldown"].get(key)
    return last is None or day - int(last) >= days or day < int(last)


def _touch(ps: Dict[str, Any], key: str, day: int) -> None:
    cd = ps["cooldown"]
    cd[key] = int(day)
    if len(cd) > 1500:
        for k in list(cd.keys())[:500]:
            cd.pop(k, None)


def persona(F: "sfe._Pass", pid: str) -> Optional[Dict[str, Any]]:
    """Stable persona for a player (archetype, handle, chaos) plus today's mood."""
    pid = str(pid or "")
    ent = _entities(F.session).get(pid)
    if not isinstance(ent, dict):
        return None
    ps = _ps(F.st)
    base = ps["personas"].get(pid)
    pers = ent.get("personality") or {}
    g = lambda k, d=50.0: float(pers.get(k, d) or d)  # noqa: E731
    if base is not None and base.get("v") != 2:
        base = None
    if base is None:
        r = random.Random(sfe._h(F.st.get("salt"), "persona", pid))
        vol, ego, prof, savvy = g("volatility"), g("ego"), g("professionalism"), g("media_savvy")
        soc, fam, comp = g("sociability"), g("family_orientation"), g("competitiveness")
        age = int(ent.get("age") or 26)
        chaos = _clip((vol * 0.62 + ego * 0.42 - prof * 0.45 - savvy * 0.25 + 18 + r.uniform(-8, 8)) / 100.0, 0.0, 1.0)
        if chaos >= 0.72:
            arch = "loose_cannon"
        elif soc < 22:
            arch = "lurker" if r.random() < 0.6 else "cryptic"
        elif (prof >= 76 and savvy >= 55) or savvy >= 86:
            arch = "corporate"
        elif fam >= 70 and age >= 26:
            arch = "family_man"
        elif soc >= 68 and age <= 25:
            arch = "meme_lord" if r.random() < 0.6 else "hype_man"
        elif soc >= 66 and ego >= 50:
            arch = "hype_man"
        elif ego >= 62 and soc >= 50:
            arch = "lifestyle"
        elif comp >= 75 and soc < 55:
            arch = "grinder"
        else:
            arch = r.choice(["grinder", "corporate", "lifestyle", "hype_man", "family_man", "cryptic"])
        social = ent.get("social") or {}
        base = {
            "v": 2,
            "arch": arch,
            "chaos": round(chaos, 3),
            "handle": str(social.get("handle") or f"@{sfe._slug(ent.get('player_name'))}{(sfe._h(pid) % 89) + 10}"),
            "dog": r.choice(DOG_NAMES),
            "emoji": r.choice(["🔥", "🚨", "🫡", "🙏", "💪", "🏒", "😤", "🤝", "👀", "🧊"]),
        }
        # Secret burner: volatile egos, plus the odd quiet guy nobody would suspect.
        if (chaos >= 0.6 and ego >= 60 and r.random() < 0.35) or r.random() < 0.006:
            burners = ps["burners"]
            taken = {b.get("handle") for b in burners.values()}
            free = [h for h in BURNER_HANDLES if h not in taken]
            if free and len([b for b in burners.values() if not b.get("unmasked")]) < 22:
                burners[pid] = {"handle": r.choice(free), "posts": 0, "suspicion": 0.0, "evidence": [], "accused": "",
                                "accused_id": "", "unmasked": False, "thread": False, "created_iso": F.iso}
        ps["personas"][pid] = base
    st_ = ent.get("state") or {}
    social = ent.get("social") or {}
    morale = float(st_.get("morale", 60) or 60)
    stress = float(st_.get("media_stress", 40) or 40)
    heat = _clip(base["chaos"] + (55 - morale) / 160.0 + (stress - 50) / 260.0, 0.0, 1.0)
    return {
        **base,
        "pid": pid,
        "name": str(ent.get("player_name") or ""),
        "last": sfe._last(str(ent.get("player_name") or "")),
        "team_id": str(ent.get("team_id") or ""),
        "followers": int(social.get("followers") or 5000),
        "morale": morale,
        "coach_trust": float(st_.get("coach_trust", 60) or 60),
        "gm_trust": float(st_.get("gm_trust", 60) or 60),
        "role_sat": float(st_.get("role_satisfaction", 60) or 60),
        "stress": stress,
        "heat": round(heat, 3),
        "banned": int(social.get("posting_ban_until", -1) or -1) >= F.day_idx,
        "ovr": float(ent.get("overall") or 0),
        "pos": str(ent.get("position") or ""),
        "community": float((ent.get("life") or {}).get("community_connection", 40) or 40),
    }


def _acct(F: "sfe._Pass", pp: Dict[str, Any]) -> Dict[str, Any]:
    tm = F.team(pp["team_id"])
    return {"id": f"player_{pp['pid']}", "name": pp["name"], "handle": pp["handle"], "type": "player", "badge": "player",
            "team_id": pp["team_id"], "color": tm.get("color"), "verified": True, "player_id": pp["pid"]}


def _avatar(F: "sfe._Pass", pid: str, pp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = F.player(pid)
    bits = sfe._headshot_bits(p) if p is not None else {}
    out = {**bits, "player_id": str(pid)}
    if pp:
        out.update({"name": pp.get("name"), "position": pp.get("pos"), "team_id": pp.get("team_id")})
    return out


def _decorate(F: "sfe._Pass", post: Optional[Dict[str, Any]], pp: Dict[str, Any], *, ratio: bool = False) -> Optional[Dict[str, Any]]:
    if not post:
        return None
    post["author_avatar"] = _avatar(F, pp["pid"], pp)
    post["persona_label"] = PERSONA_LABELS.get(pp["arch"], "")
    post["followers"] = pp["followers"]
    post["cat"] = "players"
    if ratio:
        post["replies"] = int(post.get("likes", 0) * F.rand("ratio", post["id"]).uniform(1.3, 2.6)) + 40
        post["ratioed"] = True
    return post


def _player_post(F: "sfe._Pass", pp: Dict[str, Any], text: Optional[str], *, kind: str, when: Tuple[str, str],
                 sentiment: float = 0.3, controversy: float = 0.1, mag: float = 1.0, ratio: bool = False,
                 extra_teams: List[str] = (), related: str = "") -> Optional[Dict[str, Any]]:
    if not text:
        return None
    star = F.star({"ovr": pp["ovr"] or 72})
    fmag = mag * (0.45 + min(2.6, pp["followers"] / 900_000.0))
    post = sfe._add_post(F, _acct(F, pp), text, kind=kind, cat="players", when=when,
                         team_ids=[pp["team_id"], *extra_teams], player_id=pp["pid"], player_name=pp["name"],
                         mag=fmag, star=star, sentiment=sentiment, controversy=controversy, knowledge="social", related=related)
    return _decorate(F, post, pp, ratio=ratio)


def _reply_row(F: "sfe._Pass", acct_like: Dict[str, Any], text: str, rng: random.Random, *, likes: Tuple[int, int] = (20, 900),
               pid: str = "") -> Dict[str, Any]:
    row = {"author_name": acct_like.get("name"), "handle": acct_like.get("handle"), "author_type": acct_like.get("type"),
           "verified": bool(acct_like.get("verified")), "text": text, "likes": rng.randint(*likes),
           "team_id": acct_like.get("team_id") or ""}
    if pid:
        row["player_id"] = pid
        row["author_avatar"] = _avatar(F, pid)
    return row


def _add_replies(post: Optional[Dict[str, Any]], rows: List[Optional[Dict[str, Any]]]) -> None:
    if post is None:
        return
    rows = [r for r in rows if r and r.get("text")]
    if rows:
        post["thread_replies"] = (list(post.get("thread_replies") or []) + rows)[:6]


def _teammates(F: "sfe._Pass", tid: str, exclude: str, rng: random.Random, n: int = 2, chatty: bool = True) -> List[Dict[str, Any]]:
    rows = [e for e in _by_team(F).get(str(tid), []) if str(e.get("player_id")) != str(exclude)]
    rng.shuffle(rows)
    out = []
    for e in rows:
        pp = persona(F, str(e.get("player_id")))
        if not pp or pp["banned"]:
            continue
        if chatty and pp["arch"] in ("lurker",) and rng.random() < 0.85:
            continue
        out.append(pp)
        if len(out) >= n:
            break
    return out


def _incident(F: "sfe._Pass", pp: Dict[str, Any], kind: str, text: str, *, severity: int = 1, target: str = "",
              headline: str = "", summary: str = "") -> None:
    """Apply an incident's effects to the player and log it (big ones become storylines)."""
    ent = _entities(F.session).get(pp["pid"])
    if not isinstance(ent, dict):
        return
    fx = dict(INCIDENT_FX.get(kind) or INCIDENT_FX["rant"])
    scale = 1.0 + 0.25 * (severity - 1)
    state = ent.setdefault("state", {})
    for k, v in fx.items():
        if k == "fan":
            continue
        state[k] = round(_clip(float(state.get(k, 55) or 55) + v * scale), 2)
    social = ent.setdefault("social", {})
    social["fan_sentiment"] = round(_clip(float(social.get("fan_sentiment", 55) or 55) + fx.get("fan", 0) * scale), 1)
    row = {"kind": kind, "day": F.day_idx, "iso": F.iso, "text": str(text)[:220], "severity": int(severity), "target": target,
           "player_id": pp["pid"], "player_name": pp["name"], "team_id": pp["team_id"], "is_user_team": pp["team_id"] == F.utid,
           "headline": headline}
    social["incident"] = row
    if pp["team_id"] == F.utid:
        try:
            from app.sim_engine.franchise.storyline_engine import _gm_invalidate_interactions  # noqa: WPS433

            _gm_invalidate_interactions(F.session, pp["pid"])
        except Exception:
            _log.debug("meeting cache invalidation failed", exc_info=True)
    ps = _ps(F.st)
    ps["incidents"] = (ps["incidents"] + [row])[-MAX_INCIDENTS:]
    if severity >= 2 and headline:
        try:
            from app.sim_engine.franchise.storyline_engine import _u_record_storyline  # noqa: WPS433

            event = {"id": f"uve_soc_{sfe._h(pp['pid'], kind, F.iso) % 10**8:08d}", "kind": f"social_{kind}", "team_id": pp["team_id"],
                     "participants": [pp["pid"]], "player_id": pp["pid"], "player_name": pp["name"],
                     "event_tier": "major" if severity >= 3 else "developing", "tone": "negative"}
            _u_record_storyline(F.session, event=event, headline=headline, summary=summary or str(text)[:200],
                                cause_type="SOCIAL_MEDIA", category="media", heat=58 + 10 * severity, public=True)
        except Exception:
            _log.debug("social incident storyline failed", exc_info=True)
    if kind in ("unfollow", "burner_unmasked", "rant_coach", "rant_teammates"):
        p = F.player(pp["pid"])
        if p is not None:
            try:
                from app.sim_engine.franchise.storyline_engine import _ensure_player_storyline_state  # noqa: WPS433

                pst = _ensure_player_storyline_state(p)
                pst["trade_rumor_heat"] = max(0, min(100, int(pst.get("trade_rumor_heat") or 0) + (10 if kind == "unfollow" else 5)))
            except Exception:
                _log.debug("trade rumor heat bump failed", exc_info=True)


def _next_game(F: "sfe._Pass", tid: str, within: int = 2) -> Optional[Tuple[int, str]]:
    try:
        from app.sim_engine.league.schedule_generator import _safe_slot_team_id  # noqa: WPS433
    except Exception:
        return None
    by_day = getattr(F.session, "by_day", None) or {}
    for d in range(F.day_idx + 1, F.day_idx + 1 + within):
        for sl in list(by_day.get(d, []) or []):
            h, a = str(_safe_slot_team_id(sl, "home_id")), str(_safe_slot_team_id(sl, "away_id"))
            if tid in (h, a):
                return d, (a if h == tid else h)
    return None


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

WIN_LINES: Dict[str, List[str]] = {
    "hype_man": ["LETS GOOOOO 🚨🚨 {score} and {mate} was COOKING tonight 🔥", "{fan} nation you were LOUD tonight. that one's for you 🙌",
                 "two points. zero doubt. {mate} 🤝", "BIG W. {line} and I'm still buzzing 😤", "who's got it better than us?? NOBODY 🔥 {score}"],
    "corporate": ["Great team win against a good {opp} team. Proud of the group. On to the next one.", "Big two points tonight. Thanks to the fans for the energy.",
                  "Good response from the group. Lots to build on.", "Full 60 tonight. That's the standard. #{fan}"],
    "family_man": ["Win for the little ones watching at home 👶🏒 {score}", "{score} W. home in time for bedtime stories 📚",
                   "the kids picked my pregame meal. undefeated when they pick. {score} 🍝"],
    "meme_lord": ["me after {line}: 🧍 (internally screaming)", "{line} and I'm still bad at mario kart. life is balance",
                  "the {opp} group chat is quiet tonight 🤫", "W. posting this before {mate} posts the photo where I look weird", "rate my celly 1-10. wrong answers only"],
    "cryptic": ["🙏", "Grateful.", "quiet work. loud results.", "✔️"],
    "lifestyle": ["dressed for the occasion 🕴️ {score} W", "fit check: two points 🧥", "postgame steak in {city} hits different after a W 🥩"],
    "grinder": ["Work. 🏒", "Blocked shots and two points. Good night.", "{score}. that's the job.", "Good win. Back at it."],
    "loose_cannon": ["told everyone. {score}. the {opp} can cry about it 😂", "they talked all week. {score}. scoreboard 🤫",
                     "best team in the league and it's not close. I said what I said", "{opp} fans real quiet tonight huh 🤐 {score}",
                     "{line}. still waiting on that apology from the 'experts' 🙄"],
    "lurker": ["🚨", "W"],
}

BIG_NIGHT_LINES = ["3️⃣ 🎩 hats off to {city} tonight", "{line}. pinch me 🫠", "the puck was following me around tonight. not complaining 🎯",
                   "{line} and the first beer is on {mate} 🍺", "night to remember. thank you {city} ❤️"]

GOALIE_WIN_LINES = ["{saves} saves. the boys blocked everything else 🧱", "credit to the guys in front of me. they made it easy tonight",
                    "zero. 🥅🔒", "goalie union represent 🧤 {saves} saves", "sometimes you're the windshield. tonight I was the bug zapper ⚡"]

LOSS_LINES: Dict[str, List[str]] = {
    "corporate": ["Not good enough tonight. We'll be better.", "Tough one. Credit to the {opp}. Back to work tomorrow.", "We know we have more. Regroup and respond."],
    "grinder": ["Not good enough. Back to work.", "Long night. Short memory.", "Bad loss. My fault as much as anyone's."],
    "cryptic": ["...", "🙃", "tough night. long flight.", "storms don't last forever 🌧️"],
    "hype_man": ["we'll be back. believe that 💯", "not the result we wanted. {fan} fans we owe you one 🙏"],
    "family_man": ["Rough one tonight. Thank god for postgame hugs from the kids.", "Can't win them all. Home for a few days to reset."],
    "meme_lord": ["deleting the highlights app for 24 hours", "that game was a skill issue (mine)", "anyway here's a picture of my dog. {dog} does not care that we lost 🐶"],
    "lifestyle": ["not the outcome. still the outfit 🧥", "rough night. room service and a reset."],
}

RANT_LINES: Dict[str, List[str]] = {
    "refs": ["that was the worst officiated game I've seen in my LIFE. league needs to look at this. ridiculous.",
             "refs decided that game. not us. not them. the guys in stripes. embarrassing.",
             "two missed trips and a phantom hook. some of these refs should be doing beer league",
             "someone check if the refs had money on the {opp} tonight. kidding. (am I?)",
             "I've seen better officiating at my nephew's mite tournament and those refs were 14"],
    "teammates": ["some guys in that room need to look in the mirror. not saying names. they know.",
                  "you can't win when half the team shows up. I'm done pretending",
                  "same guys every night. effort is free. some of us forgot that apparently",
                  "not everybody in that room cares as much as they say they do. that's all.",
                  "I'll take my share of the blame. some guys need to take theirs."],
    "coach": ["hard to make an impact when you're watching from the bench half the third. just saying.",
              "funny how the guys who play the most aren't the ones scoring. anyway.",
              "{pts} points and I'm on the third line. make it make sense 🤔",
              "coach has his guys. I'm clearly not one of them. cool.",
              "love being the first guy benched when things go wrong. builds character apparently"],
    "fans": ["booing your own team in the 2nd period is weak. some of you were never real fans",
             "to the guy in section 112 screaming at me the whole game: buy a ticket to watch yourself next time",
             "love {city}. some of the fans tonight? not so much",
             "the people throwing stuff on the ice should be banned for life. grow up"],
    "opponent": ["{opp} celebrating a regular season win like they won the cup lol. see you in the rematch 😘",
                 "enjoy it {opp}. we'll remember this one 📝", "{opp_star} talked a lot of trash for a guy who disappears in the third"],
    "media": ["the media in this city will write whatever gets clicks. I'm done reading it",
              "can't wait for the hot takes tomorrow. stay classy local media 👏",
              "to the radio host who said I 'look disinterested': come say it in the room"],
}

CRYPTIC_LINES = ["sometimes you have to bet on yourself.", "🌅 new chapters", "loyalty is a two way street", "be where you're celebrated, not tolerated",
                 "noise.", "everything happens for a reason 🙏", "hmmm.", "funny how fast things change", "know your worth. then add tax.",
                 "👀", "patience is running thin and so is my hair", "some people only respect you when you leave. noted."]

LIFESTYLE_LINES: Dict[str, List[str]] = {
    "family_man": ["first skate for the little guy ⛸️ he already has better edges than me", "date night in {city} 🍝 babysitter MVP",
                   "{dog} has a new chew toy. it's my stick. send help", "pancake sunday. the kids want chocolate chips in everything 🥞"],
    "meme_lord": ["lost 6 straight games of FIFA to {mate}. requesting a trade (of controllers)", "if you see me at the {city} Costco no you didn't 🛒",
                  "unpopular opinion: practice jerseys should be pastel", "{mate} just asked me if Canada has a president. I'm leaving the team"],
    "lifestyle": ["fit check before the road trip 🧳", "found the best coffee in {city} ☕ not telling you where", "golf day 🏌️ shot a 94. don't @ me",
                  "new watch. same work ethic ⌚"],
    "grinder": ["early skate. empty rink. best part of the day.", "6am lift. no shortcuts.", "film. sleep. repeat."],
    "corporate": ["Great day visiting the kids at the {city} Children's Hospital. They were the real all-stars today. ❤️",
                  "Thanks to everyone who came out to the community skate today. {city} shows up. 🙏", "Proud to support local minor hockey in {city}."],
    "hype_man": ["{fan} fans you've been unreal this year. next home game LET'S BLOW THE ROOF OFF 🔥", "team dinner. big vibes. {mate} paid (he didn't know) 💸",
                 "shoutout to the equipment guys. legends behind the scenes 🫡"],
    "cryptic": ["📖", "quiet mornings.", "🌲"],
    "loose_cannon": ["pineapple belongs on pizza and so does ketchup. fight me", "hot take: the {rival} are overrated and their mascot is ugly",
                     "3 hours of sleep and a protein bar. elite.", "someone in {city} just cut me off on the highway and honked at ME. respectfully, learn to drive",
                     "ranking my teammates' haircuts. thread 🧵 (I will be traded)"],
    "lurker": [],
}

TEAMMATE_REPLIES_WIN = ["🐐", "carried us tonight", "dinner's on you", "that celly though 😂", "my guy 🤝", "🔥🔥🔥", "assist was better than the goal tbh",
                        "MVP MVP MVP", "send me the clip", "📈📈", "postgame playlist goes crazy btw", "the boys 🫡", "WHAT A NIGHT", "I'll take credit for the screen",
                        "we're framing that one", "proud of you dawg", "airplane mode until tomorrow, goodnight legends", "💯", "bus ride home is gonna be loud",
                        "you owe me for that pass", "hardest working guy in the room", "big boy game 💪", "told you it was going in", "🧊🧊🧊"]
TEAMMATE_REPLIES_LIFE = ["🤣", "invite me next time", "this is why we lose at cards", "who took this photo 💀", "I want a refund on this post", "❤️"]
FAN_REPLIES_GOOD = ["{last} for captain", "marry me (in a hockey way)", "framing this", "{last} supremacy", "best player on the team and it's not close"]
FAN_REPLIES_BAD = ["less posting more scoring", "ratio", "delete this", "this aged well", "imagine posting this at {cap}", "log off brother", "we need a trade. yours."]
FAN_REPLIES_RANT = ["he's not wrong though", "delete this before PR sees it", "absolute cinema", "the screenshot police are on the way 🚨",
                    "this man needs a media trainer and a hug", "finally someone said it", "this is going to be a whole thing isn't it"]

BURNER_LINES: Dict[str, List[str]] = {
    "self": ["people sleep on {last}. dude does everything right and gets zero credit", "{last} is the only guy on the {nick} who shows up every night. facts",
             "if {last} played in a big market he'd be a household name", "{last} quietly having a career year and nobody's talking about it",
             "the {nick} would be lost without {last}. that's not even debatable"],
    "coach": ["how is {mate_last} getting PP1 over {last}? {coach_last} has favourites and everyone in the league knows it",
              "{coach_last}'s system is from 2009. someone please tell him", "imagine benching your best player in the third. couldn't be the {nick}... oh wait"],
    "leak": ["hearing the {nick} room is split. vets vs the kids. not great", "practice got heated today. that's all I'll say 🤐",
             "{mate_last} was late to the team bus AGAIN lol", "players-only meeting after the last one. not from me obviously 🤫"],
    "fans": ["imagine booing {last} after the season he's having. clowns", "the {nick} fanbase doesn't deserve {last}. there I said it"],
    "money": ["{last} deserves a raise and everybody in that front office knows it", "{last} is massively underpaid compared to {mate_last}. look it up"],
}

BURNER_EVIDENCE = [
    "Posts within 20 minutes of {last}'s games ending, home and road",
    "Never posts while the {nick} are playing",
    "Defends {last} in {n} of {m} posts",
    "Uses the same rare emoji ({emoji}) as {last}'s main account",
    "Account was created the week {last} got to {city}",
    "Follows {last}'s brother and his old junior coach",
    "Same misspelling ('definately') shows up in {last}'s old posts",
    "Knew about the bus being late before any reporter did",
    "Went completely silent the week {last} was hurt",
    "Liked a post from {last}'s girlfriend at 2:14 AM, then unliked it",
]


def _f(tmpl: str, ctx: Dict[str, Any]) -> Optional[str]:
    return sfe._fill(tmpl, ctx)


def _fresh(F: "sfe._Pass", rng: random.Random, pool: List[str], ctx: Dict[str, Any]) -> Optional[str]:
    filled = [t for t in (_f(p, ctx) for p in pool) if t]
    if not filled:
        return None
    return sfe._pick_fresh(F.st, rng, filled)


# ---------------------------------------------------------------------------
# Played-day generator
# ---------------------------------------------------------------------------


def gen_player_voices(F: "sfe._Pass", games: List[Dict[str, Any]], lines: Dict[str, Dict[str, Any]]) -> None:
    ps = _ps(F.st)
    if ps.get("season") != F.season:
        ps["season"] = F.season
        ps["guarantees"] = []
    budget = {"posts": 26, "rants": 3, "life": 7, "cryptic": 3, "burner": 3, "likes": 1, "unfollows": 1}
    playing: set = set()
    for g in games:
        playing.update({str(g.get("home_id")), str(g.get("away_id"))})
        try:
            _game_voices(F, g, lines, budget)
        except Exception:
            _log.exception("player game voices failed")
    for fn in (_resolve_guarantees, _mood_voices, _make_guarantees, _lifestyle_voices, _burner_voices, gen_player_event_voices):
        try:
            if fn is _lifestyle_voices:
                fn(F, budget, playing)
            elif fn is _resolve_guarantees:
                fn(F, games)
            elif fn is gen_player_event_voices:
                fn(F)
            else:
                fn(F, budget)
        except Exception:
            _log.exception("player social generator %s failed", getattr(fn, "__name__", fn))


def _game_voices(F: "sfe._Pass", g: Dict[str, Any], lines: Dict[str, Dict[str, Any]], budget: Dict[str, int]) -> None:
    hid, aid = str(g.get("home_id")), str(g.get("away_id"))
    if hid not in F.teams or aid not in F.teams:
        return
    hs = sfe._si(g.get("home_score", g.get("home_goals")))
    as_ = sfe._si(g.get("away_score", g.get("away_goals")))
    win, lose = (hid, aid) if hs > as_ else (aid, hid)
    ws, ls = max(hs, as_), min(hs, as_)
    rng = F.rand("pvoice", g.get("game_id"))
    user_game = F.utid in (hid, aid)
    team_lines = {t: sorted([d for d in lines.values() if d.get("team_id") == t], key=lambda d: (d.get("pts", 0), d.get("g", 0), d.get("saves", 0)), reverse=True)
                  for t in (win, lose)}
    for tid in (win, lose):
        if budget["posts"] <= 0:
            return
        won = tid == win
        tm, opp = F.team(tid), F.team(lose if won else win)
        mine = team_lines.get(tid) or []
        cands: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for d in mine:
            pp = persona(F, d["id"])
            if not pp or pp["banned"]:
                continue
            gk = str(d.get("pos") or "").startswith("G")
            big = (d.get("pts", 0) >= 3 or d.get("g", 0) >= 2) if not gk else (d.get("w") and (d.get("saves", 0) >= 32 or d.get("ga", 9) == 0))
            weight = PERSONA_ACTIVITY.get(pp["arch"], 0.8)
            if won:
                weight *= (2.6 if big else 1.0 if d.get("pts", 0) else 0.35)
            else:
                weight *= 0.5 + pp["heat"] * (2.2 if ws - ls >= 3 else 1.3)
            if tm.get("is_user"):
                weight *= 1.6
            cands.append((weight * rng.uniform(0.4, 1.3), pp, d))
        cands.sort(key=lambda t: -t[0])
        n_posts = 2 if (tm.get("is_user") or user_game) else 1
        for weight, pp, d in cands[:n_posts]:
            if weight < (0.35 if tm.get("is_user") else 0.9) or budget["posts"] <= 0:
                continue
            if won:
                _win_post(F, rng, pp, d, tm, opp, ws, ls, mine)
            else:
                _loss_post(F, rng, pp, d, tm, opp, ws, ls, budget, g)
            budget["posts"] -= 1


def _ctx_for(F: "sfe._Pass", pp: Dict[str, Any], tm: Dict[str, Any], opp: Dict[str, Any], ws: int, ls: int, d: Dict[str, Any],
             mate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {"score": f"{ws}-{ls}", "opp": opp.get("nick"), "fan": tm.get("fan"), "city": tm.get("city"), "line": sfe._game_line_text(d) if d.get("pts") else None,
            "mate": mate["handle"] if mate else None, "saves": d.get("saves") or None, "dog": pp.get("dog"), "last": pp["last"],
            "rec": F.rec(tm.get("id")).get("rec")}


def _win_post(F, rng, pp, d, tm, opp, ws, ls, mine) -> None:
    mate_line = next((m for m in mine if m["id"] != d["id"] and m.get("pts", 0) > 0), None)
    mate = persona(F, mate_line["id"]) if mate_line else None
    ctx = _ctx_for(F, pp, tm, opp, ws, ls, d, mate)
    gk = str(d.get("pos") or "").startswith("G")
    if gk:
        pool = GOALIE_WIN_LINES if pp["arch"] not in ("cryptic", "lurker", "grinder") else WIN_LINES[pp["arch"]]
    elif d.get("g", 0) >= 3 or d.get("pts", 0) >= 4:
        pool = BIG_NIGHT_LINES + WIN_LINES.get(pp["arch"], [])[:2]
    else:
        pool = WIN_LINES.get(pp["arch"], WIN_LINES["grinder"])
    text = _fresh(F, rng, pool, ctx)
    when = F.stamp(F.iso, 22 * 60 + 25, 23 * 60 + 58, rng)
    post = _player_post(F, pp, text, kind="player_post", when=when, sentiment=0.7, mag=1.2 if d.get("pts", 0) >= 3 else 0.9)
    if not post:
        return
    replies = []
    for mt in _teammates(F, pp["team_id"], pp["pid"], rng, n=rng.randint(1, 3)):
        replies.append(_reply_row(F, _acct(F, mt), sfe._pick_fresh(F.st, rng, TEAMMATE_REPLIES_WIN), rng, pid=mt["pid"]))
    fan = sfe._fan_account(F, pp["team_id"], "homer", rng)
    replies.append(_reply_row(F, fan, _f(rng.choice(FAN_REPLIES_GOOD), {"last": pp["last"]}) or "", rng, likes=(5, 300)))
    _add_replies(post, replies)
    # Loose cannons and hype men chirp the other side; a volatile opponent may bite.
    if pp["arch"] in ("loose_cannon", "hype_man") and pp["heat"] >= 0.45 and rng.random() < 0.5:
        opp_rows = [e for e in _by_team(F).get(str(opp.get("id")), [])]
        biters = [persona(F, str(e.get("player_id"))) for e in opp_rows]
        biters = [b for b in biters if b and b["heat"] >= 0.5 and not b["banned"]]
        if biters:
            b = max(biters, key=lambda x: x["heat"])
            clap = sfe._pick_fresh(F.st, rng, [f"enjoy it. see you soon {pp['handle']} 📝", "talk is cheap. so is that goal",
                                               f"{pp['last']} posting like he won the cup lol", "screenshotting this for later 📸",
                                               "relax it's one game in January brother", "bold words from a guy who was -2 last time we played"])
            _add_replies(post, [_reply_row(F, _acct(F, b), clap, rng, likes=(400, 9000), pid=b["pid"])])
            post["beef"] = {"with_player_id": b["pid"], "with_name": b["name"]}
            post["controversy"] = 0.7
            if rng.random() < 0.5:
                meme = sfe._meme_account(rng)
                sfe._add_post(F, meme, sfe._pick_fresh(F.st, rng, [f"{pp['last']} and {b['last']} are fighting in the replies and I have never been more locked in 🍿",
                              f"{b['last']} really said that to {pp['last']} in the replies. next game is must-watch TV 📺",
                              f"the {pp['last']} vs {b['last']} reply war is the best rivalry in hockey right now and it's happening on this app",
                              f"somebody get {pp['last']} and {b['last']} in a ring. or a podcast. either works 🎙️"]),
                              kind="beef", cat="players", when=F.stamp(F.iso, 23 * 60 + 20, 23 * 60 + 59, rng),
                              team_ids=[pp["team_id"], b["team_id"]], player_id=pp["pid"], player_name=pp["name"], mag=1.3, controversy=0.6)


def _loss_post(F, rng, pp, d, tm, opp, ws, ls, budget, g) -> None:
    ctx = _ctx_for(F, pp, tm, opp, ws, ls, d, None)
    # Rants: hot heads after a bad loss, or a guy whose ice time got cut.
    toi = int((d.get("toi_sec") or 0) / 60) if d.get("toi_sec") else 0
    p_rant = pp["heat"] ** 1.6 * (1.4 if ws - ls >= 3 else 1.0) * (0.25 if pp["arch"] == "corporate" else 1.0)
    if pp["arch"] == "loose_cannon" or (pp["heat"] >= 0.55 and rng.random() < 0.5):
        if budget["rants"] > 0 and rng.random() < p_rant:
            budget["rants"] -= 1
            _rant(F, rng, pp, tm, opp, ws, ls, toi, g)
            return
    pool = LOSS_LINES.get(pp["arch"]) or LOSS_LINES["grinder"]
    text = _fresh(F, rng, pool, ctx)
    post = _player_post(F, pp, text, kind="player_post", when=F.stamp(F.iso, 22 * 60 + 40, 23 * 60 + 59, rng), sentiment=-0.3, mag=0.7,
                        ratio=bool(pp["arch"] in ("meme_lord", "lifestyle") and rng.random() < 0.4))
    if post and rng.random() < 0.6:
        fan = sfe._fan_account(F, pp["team_id"], "doomer", rng)
        cap = sfe._money(F.pinfo(pp["pid"]).get("cap") or 0) if F.pinfo(pp["pid"]).get("cap") else "this salary"
        _add_replies(post, [_reply_row(F, fan, _f(rng.choice(FAN_REPLIES_BAD), {"cap": cap}) or "log off", rng, likes=(10, 700))])


def _rant(F, rng, pp, tm, opp, ws, ls, toi, g) -> None:
    targets = ["refs", "opponent", "media"]
    if pp["coach_trust"] < 48 or pp["role_sat"] < 45:
        targets += ["coach", "coach"]
    if ws - ls >= 3:
        targets += ["teammates"]
    if tm.get("market", 1.0) >= 1.15 and str(g.get("home_id")) == pp["team_id"]:
        targets += ["fans"]
    target = rng.choice(targets)
    opp_star = None
    opp_rows = sorted(_by_team(F).get(str(opp.get("id")), []), key=lambda e: -float(e.get("overall") or 0))
    if opp_rows:
        opp_star = sfe._last(str(opp_rows[0].get("player_name") or ""))
    season_pts = F.pinfo(pp["pid"]).get("pts") or None
    ctx = {"opp": opp.get("nick"), "city": tm.get("city"), "pts": season_pts if season_pts and season_pts >= 6 else None, "opp_star": opp_star}
    text = _fresh(F, rng, RANT_LINES[target], ctx)
    if not text:
        return
    if pp["heat"] >= 0.85 and rng.random() < 0.2:
        text = text.upper()
    nxt = sfe._iso_shift(F.iso, 1)
    when = F.stamp(nxt, 60 + rng.randint(0, 40), 3 * 60 + 10, rng)
    post = _player_post(F, pp, text, kind="player_rant", when=when, sentiment=-0.8, controversy=0.9, mag=1.6, ratio=rng.random() < 0.5)
    if not post:
        return
    fans = [sfe._fan_account(F, pp["team_id"], pers, rng) for pers in ("ironic", "doomer")]
    _add_replies(post, [_reply_row(F, fans[0], sfe._pick_fresh(F.st, rng, FAN_REPLIES_RANT), rng, likes=(80, 3000)),
                        _reply_row(F, fans[1], sfe._pick_fresh(F.st, rng, FAN_REPLIES_RANT), rng, likes=(40, 2000))])
    deleted = rng.random() < (0.35 + (0.3 if target in ("coach", "teammates", "fans") else 0.0))
    severity = 1 + int(target in ("coach", "teammates", "fans")) + int(pp["followers"] > 900_000 and target in ("coach", "teammates"))
    if deleted:
        mins = rng.randint(6, 74)
        post["deleted"] = True
        post["deleted_note"] = f"Deleted after {mins} minutes"
        arch = sfe._pick(rng, SLEUTH_ACCOUNTS[2:3] + [sfe._meme_account(rng)])
        acct = {**arch, "type": "meme", "badge": "meme", "verified": False}
        shot = sfe._add_post(F, acct, sfe._pick_fresh(F.st, rng, [
            f"{pp['name']} posted this at {post['time']} and deleted it {mins} minutes later. we have screenshots. we always have screenshots 📸",
            f"the internet never forgets, {pp['last']} 📸",
            f"{pp['last']} with the {post['time']} post. deleted. archived. framed. 🖼️"]),
            kind="screenshot", cat="players", when=F.stamp(nxt, 3 * 60 + 15, 8 * 60, rng), team_ids=[pp["team_id"]],
            player_id=pp["pid"], player_name=pp["name"], mag=1.8, controversy=0.8)
        if shot:
            shot["quote"] = {"handle": pp["handle"], "name": pp["name"], "text": text, "time": post["time"], "deleted": True,
                             "author_avatar": post.get("author_avatar")}
    kind = {"coach": "rant_coach", "teammates": "rant_teammates", "fans": "rant_fans"}.get(target, "rant")
    tgt_name = {"coach": tm.get("coach") or "the coach", "teammates": "his teammates", "fans": "the fans"}.get(target, target)
    _incident(F, pp, kind, text, severity=severity, target=target,
              headline=f"{pp['name']} {'deletes' if deleted else 'posts'} late-night shot at {tgt_name}" if severity >= 2 else "",
              summary=f"After a {ws}-{ls} loss to the {opp.get('nick')}, {pp['name']} posted: “{text[:140]}”" + (" It was deleted, but not before the screenshots." if deleted else ""))
    if severity >= 2:
        beat = sfe._beat_writer(F, pp["team_id"])
        coach = tm.get("coach") or "the coach"
        fallout = (sfe._pick_fresh(F.st, rng, [
            f"Asked {coach} about {pp['last']}'s post this morning: \"I haven't seen it. We'll handle it internally.\" Expect that to be the line all day.",
            f"{coach} on {pp['last']}'s overnight post: \"Everyone's entitled to an opinion. Mine decides the lineup.\" Ice cold.",
            f"{pp['last']} was not on the ice for the start of practice this morning. Coach called it 'a conversation'. Hmm.",
            f"{coach}, asked if he and {pp['last']} are on the same page: long pause. \"We're fine.\" Next question."])
            if target == "coach" else sfe._pick_fresh(F.st, rng, [
            f"{pp['last']}'s overnight post is the talk of the room this morning. Teammates declined to comment. That says plenty.",
            f"Captain on {pp['last']}'s post: \"We'll talk about it as a group. Not here.\"",
            f"Tense room this morning after {pp['last']}'s post. Music was off in the dressing room. Usually it isn't.",
            f"{pp['last']} met with reporters and said his post 'came out wrong'. Didn't say what he meant instead."]))
        sfe._add_post(F, beat, fallout,
                      kind="rant_fallout", cat="players", when=F.stamp(nxt, 9 * 60 + 30, 12 * 60, rng), team_ids=[pp["team_id"]],
                      player_id=pp["pid"], player_name=pp["name"], mag=1.3, controversy=0.6, knowledge="report")


# ---------------------------------------------------------------------------
# Mood, likes, unfollows, guarantees, lifestyle
# ---------------------------------------------------------------------------


def _all_personas(F: "sfe._Pass") -> List[Dict[str, Any]]:
    out = []
    for tid, rows in _by_team(F).items():
        if tid not in F.teams:
            continue
        for e in rows:
            pp = persona(F, str(e.get("player_id")))
            if pp:
                out.append(pp)
    return out


def _mood_voices(F: "sfe._Pass", budget: Dict[str, int]) -> None:
    ps = _ps(F.st)
    rng = F.rand("pmood")
    pool = [pp for pp in _all_personas(F) if not pp["banned"]]
    rng.shuffle(pool)
    for pp in pool:
        unhappy = pp["morale"] < 38 or pp["role_sat"] < 38 or pp["gm_trust"] < 35 or pp["coach_trust"] < 35
        if not unhappy:
            continue
        tm = F.team(pp["team_id"])
        # Unfollow: once a season, only when really fed up.
        if (budget["unfollows"] > 0 and pp["morale"] < 30 and pp["gm_trust"] < 40 and pp["heat"] >= 0.5 and rng.random() < 0.05
                and _cooldown_ok(ps, f"unf:{pp['pid']}", F.day_idx, 400) and _flag_season(F, f"unf:{pp['pid']}")):
            _touch(ps, f"unf:{pp['pid']}", F.day_idx)
            budget["unfollows"] -= 1
            _unfollow(F, rng, pp, tm)
            continue
        # Liking a post that calls for the coach's head.
        if (budget["likes"] > 0 and pp["coach_trust"] < 42 and pp["heat"] >= 0.45 and tm.get("coach") and rng.random() < 0.03
                and _cooldown_ok(ps, f"like:{pp['pid']}", F.day_idx, 30)):
            _touch(ps, f"like:{pp['pid']}", F.day_idx)
            budget["likes"] -= 1
            _like_event(F, rng, pp, tm)
            continue
        if pp["arch"] in ("lurker", "grinder") and rng.random() < 0.75:
            continue
        if budget["cryptic"] > 0 and pp["arch"] not in ("corporate",) and rng.random() < 0.06 + pp["heat"] * 0.12 \
                and _cooldown_ok(ps, f"cry:{pp['pid']}", F.day_idx, 8):
            _touch(ps, f"cry:{pp['pid']}", F.day_idx)
            budget["cryptic"] -= 1
            text = sfe._pick_fresh(F.st, rng, CRYPTIC_LINES)
            post = _player_post(F, pp, text, kind="player_cryptic", when=F.stamp(F.iso, 13 * 60, 23 * 60 + 30, rng), sentiment=-0.2, controversy=0.5, mag=1.2)
            if post:
                post["mood_hint"] = "unhappy"
                _incident(F, pp, "cryptic", text, severity=1)
                if pp["ovr"] >= 78 or pp["team_id"] == F.utid:
                    sfe._add_thread(F, sub=tm.get("sub") or "r/hockey", title=f"{pp['name']} just posted \"{text}\" — are we reading into this?",
                                    body=f"Posted at {post['time']}. No caption, no context. {pp['last']}'s been {('frustrated with his role' if pp['role_sat'] < 45 else 'quiet')} lately.",
                                    kind="cryptic", cat="players", flair="Speculation", when=F.stamp(F.iso, 14 * 60, 23 * 60 + 50, rng),
                                    comments=[sfe._comment(sfe._reddit_user(rng, tm), "he's gone by the deadline. calling it now", rng.randint(80, 600), sent=-0.6),
                                              sfe._comment(sfe._reddit_user(rng, tm), "or he just likes sunsets. relax", rng.randint(60, 400), sent=0.2),
                                              sfe._comment(sfe._reddit_user(rng), "the 'loyalty is a two way street' arc is never good news", rng.randint(20, 200), sent=-0.3)],
                                    team_ids=[pp["team_id"]], player_id=pp["pid"], player_name=pp["name"], mag=1.1, sentiment=-0.3, knowledge="speculation")


def _flag_season(F: "sfe._Pass", key: str) -> bool:
    k = f"{key}:{F.season}"
    if sfe._flag(F.st, k):
        return False
    sfe._set_flag(F.st, k)
    return True


def _unfollow(F, rng, pp, tm) -> None:
    sl = SLEUTH_ACCOUNTS[0] if rng.random() < 0.5 else SLEUTH_ACCOUNTS[1]
    acct = {**sl, "type": "meme", "badge": "sleuth", "verified": False}
    text = f"👀 {pp['name']} has unfollowed the official {tm.get('nick')} account and removed \"{tm.get('full')}\" from his bio. Make of that what you will."
    post = sfe._add_post(F, acct, text, kind="unfollow", cat="players", when=F.stamp(F.iso, 10 * 60, 22 * 60, rng), team_ids=[pp["team_id"]],
                         player_id=pp["pid"], player_name=pp["name"], mag=2.2, controversy=0.9, knowledge="claim")
    if post:
        post["evidence_card"] = {"type": "bio_diff", "before": f"{tm.get('full')} | {pp['pos'] or 'F'}", "after": "🏒", "handle": pp["handle"],
                                 "author_avatar": _avatar(F, pp["pid"], pp)}
    sfe._add_thread(F, sub=tm.get("sub") or "r/hockey", title=f"{pp['name']} unfollowed the team and scrubbed his bio",
                    body=f"Bio used to say \"{tm.get('full')}\". Now it's just a hockey stick emoji. He's been unhappy for weeks.",
                    kind="unfollow", cat="players", flair="Rumour", when=F.stamp(F.iso, 11 * 60, 23 * 60, rng),
                    comments=[sfe._comment(sfe._reddit_user(rng, tm), "this is how it starts. trade request incoming", rng.randint(200, 1400), sent=-0.6),
                              sfe._comment(sfe._reddit_user(rng, tm), "GM needs to get in a room with him TODAY", rng.randint(150, 900), sent=-0.3),
                              sfe._comment(sfe._reddit_user(rng), "the 'scrubbed bio' is the hockey equivalent of changing your relationship status", rng.randint(80, 600), sent=0.0),
                              sfe._comment(sfe._reddit_user(rng, tm), "maybe his phone just did it by itself 🙃", rng.randint(40, 300), sent=0.1)],
                    team_ids=[pp["team_id"]], player_id=pp["pid"], player_name=pp["name"], mag=2.0, sentiment=-0.5, knowledge="claim")
    _incident(F, pp, "unfollow", text, severity=2, target="team",
              headline=f"{pp['name']} unfollows team account, scrubs bio",
              summary=f"{pp['name']} removed the {tm.get('nick')} from his social bio and unfollowed the club. Around the league it's being read as a player who wants out.")


def _like_event(F, rng, pp, tm) -> None:
    coach_last = sfe._last(tm.get("coach") or "the coach")
    fan_text = rng.choice([f"fire {coach_last} into the sun", f"{coach_last} has lost this room. everyone can see it",
                           f"{pp['last']} deserves a coach who actually uses him", f"{coach_last}'s line blender is a war crime"])
    acct = {**SLEUTH_ACCOUNTS[3 if rng.random() < 0.3 else 2], "type": "meme", "badge": "sleuth", "verified": False}
    post = sfe._add_post(F, acct, f"{pp['name']} liked this post at {rng.randint(1, 3)}:{rng.randint(10, 59)} AM. It has since been unliked. 👀",
                         kind="liked", cat="players", when=F.stamp(F.iso, 8 * 60, 12 * 60, rng), team_ids=[pp["team_id"]],
                         player_id=pp["pid"], player_name=pp["name"], mag=1.6, controversy=0.7, knowledge="claim")
    if post:
        fan = sfe._fan_account(F, pp["team_id"], "doomer", rng)
        post["quote"] = {"handle": fan.get("handle"), "name": fan.get("name"), "text": fan_text, "liked_by": pp["name"],
                         "liked_by_avatar": _avatar(F, pp["pid"], pp)}
    _incident(F, pp, "like", fan_text, severity=2 if pp["followers"] > 700_000 or pp["team_id"] == F.utid else 1, target="coach",
              headline=f"{pp['name']} caught liking post calling for {coach_last}'s job",
              summary=f"{pp['name']} liked, then unliked, a post that read “{fan_text}”.")


def _make_guarantees(F: "sfe._Pass", budget: Dict[str, int]) -> None:
    ps = _ps(F.st)
    rng = F.rand("guarantee")
    open_g = {g["pid"] for g in ps["guarantees"] if not g.get("resolved")}
    for pp in _all_personas(F):
        if pp["arch"] not in ("loose_cannon", "hype_man") or pp["banned"] or pp["pid"] in open_g:
            continue
        r = F.rec(pp["team_id"])
        if not r or r.get("gp", 0) < 5:
            continue
        slump = r.get("l10") and int(str(r.get("l10")).split("-")[0] or 5) <= 3
        if not (slump or pp["heat"] >= 0.7) or rng.random() > 0.035 + 0.05 * pp["heat"]:
            continue
        nxt = _next_game(F, pp["team_id"], within=2)
        if not nxt:
            continue
        day, opp_id = nxt
        opp = F.team(opp_id)
        text = sfe._pick_fresh(F.st, rng, [f"we're beating the {opp.get('nick')}. write it down. 🔒", f"{opp.get('nick')} next. guaranteed W. screenshot this.",
                                           "I don't care what the standings say. we win the next one. guarantee it.",
                                           f"gonna say it now so nobody can say I didn't: we beat {opp.get('abbr')}. 🔒"])
        post = _player_post(F, pp, text, kind="guarantee", when=F.stamp(F.iso, 15 * 60, 22 * 60, rng), sentiment=0.4, controversy=0.6, mag=1.6, extra_teams=[opp_id])
        if post:
            ps["guarantees"].append({"pid": pp["pid"], "team_id": pp["team_id"], "opp_id": opp_id, "day": day, "post_id": post["id"], "resolved": False})
            ps["guarantees"] = ps["guarantees"][-30:]
            beat = sfe._beat_writer(F, opp_id)
            sfe._add_post(F, beat, f"{opp.get('abbr')} room has seen {pp['last']}'s guarantee. One player, smiling: \"Bulletin board material. Thanks.\"",
                          kind="guarantee_react", cat="players", when=F.stamp(F.iso, 22 * 60 + 5, 23 * 60 + 50, rng), team_ids=[opp_id, pp["team_id"]], mag=1.0)
            budget["posts"] -= 1
            return  # one called shot a day is plenty


def _resolve_guarantees(F: "sfe._Pass", games: List[Dict[str, Any]]) -> None:
    ps = _ps(F.st)
    for gu in ps["guarantees"]:
        if gu.get("resolved") or int(gu.get("day", -1)) > F.day_idx:
            continue
        game = next((g for g in games if {str(g.get("home_id")), str(g.get("away_id"))} == {gu["team_id"], gu["opp_id"]}), None)
        if game is None:
            if int(gu.get("day", -1)) < F.day_idx - 2:
                gu["resolved"] = True
            continue
        gu["resolved"] = True
        pp = persona(F, gu["pid"])
        if not pp:
            continue
        rng = F.rand("gres", gu["pid"])
        hs, as_ = sfe._si(game.get("home_score")), sfe._si(game.get("away_score"))
        mine = hs if str(game.get("home_id")) == gu["team_id"] else as_
        theirs = as_ if str(game.get("home_id")) == gu["team_id"] else hs
        opp = F.team(gu["opp_id"])
        if mine > theirs:
            _player_post(F, pp, sfe._pick_fresh(F.st, rng, ["TOLD YOU 🔒", "called it. 🔒 who's laughing now", f"guaranteed it. delivered it. {mine}-{theirs}. goodnight {opp.get('nick')} 😴"]),
                         kind="guarantee_kept", when=F.stamp(F.iso, 22 * 60 + 35, 23 * 60 + 59, rng), sentiment=0.9, mag=1.8, extra_teams=[gu["opp_id"]])
            sfe._add_post(F, sfe._insider("hart"), f"{pp['name']} called his shot and delivered. {mine}-{theirs} over the {opp.get('nick')}. You love to see it. Or you hate to see it. Either way you see it.",
                          kind="guarantee_kept", cat="players", when=F.stamp(F.iso, 22 * 60 + 50, 23 * 60 + 59, rng), team_ids=[gu["team_id"]],
                          player_id=pp["pid"], player_name=pp["name"], mag=1.4)
            _incident(F, pp, "guarantee_win", "Guarantee delivered", severity=1)
        else:
            receipt = sfe._add_post(F, sfe._meme_account(rng), f"{pp['name']} guaranteed a win against the {opp.get('nick')}. Final: {theirs}-{mine}. The receipts are eternal 🧾",
                                    kind="guarantee_failed", cat="players", when=F.stamp(F.iso, 22 * 60 + 30, 23 * 60 + 59, rng), team_ids=[gu["team_id"], gu["opp_id"]],
                                    player_id=pp["pid"], player_name=pp["name"], mag=2.0, controversy=0.7)
            if receipt:
                orig = next((p for p in reversed(list(getattr(F.session, "social_posts", None) or [])) if p.get("id") == gu.get("post_id")), None)
                if orig:
                    receipt["quote"] = {"handle": orig.get("handle"), "name": orig.get("author_name"), "text": orig.get("text"), "time": orig.get("time"),
                                        "author_avatar": orig.get("author_avatar")}
            _incident(F, pp, "guarantee_fail", "Guarantee failed", severity=2 if gu["team_id"] == F.utid else 1,
                      headline=f"{pp['name']}'s guarantee blows up in a loss to the {opp.get('nick')}",
                      summary=f"{pp['name']} guaranteed a win. The {opp.get('nick')} won {theirs}-{mine}.")


def _lifestyle_voices(F: "sfe._Pass", budget: Dict[str, int], playing: set) -> None:
    rng = F.rand("plife")
    ps = _ps(F.st)
    teams = [t for t in F.teams if t not in playing]
    rng.shuffle(teams)
    if F.utid in teams:
        teams.remove(F.utid)
        teams.insert(0, F.utid)
    rivals = [t for t in F.teams if F.team(t).get("div") and t != F.utid]
    for tid in teams:
        if budget["life"] <= 0:
            return
        if tid != F.utid and rng.random() > 0.35:
            continue
        rows = [persona(F, str(e.get("player_id"))) for e in _by_team(F).get(tid, [])]
        rows = [p for p in rows if p and not p["banned"] and LIFESTYLE_LINES.get(p["arch"]) and _cooldown_ok(ps, f"life:{p['pid']}", F.day_idx, 5)]
        if not rows:
            continue
        rows.sort(key=lambda p: -PERSONA_ACTIVITY.get(p["arch"], 0.5) * rng.uniform(0.3, 1.5))
        pp = rows[0]
        if pp["arch"] == "corporate" and pp["community"] < 55:
            continue
        tm = F.team(tid)
        mates = _teammates(F, tid, pp["pid"], rng, n=2)
        rival = F.team(rng.choice(rivals)).get("nick") if rivals else "Leafs"
        ctx = {"city": tm.get("city"), "fan": tm.get("fan"), "dog": pp["dog"], "mate": mates[0]["handle"] if mates else None, "rival": rival}
        text = _fresh(F, rng, LIFESTYLE_LINES[pp["arch"]], ctx)
        post = _player_post(F, pp, text, kind="player_lifestyle", when=F.stamp(F.iso, 9 * 60, 21 * 60, rng), sentiment=0.5, mag=0.8,
                            controversy=0.4 if pp["arch"] == "loose_cannon" else 0.05)
        if not post:
            continue
        _touch(ps, f"life:{pp['pid']}", F.day_idx)
        budget["life"] -= 1
        _add_replies(post, [_reply_row(F, _acct(F, m), sfe._pick_fresh(F.st, rng, TEAMMATE_REPLIES_LIFE), rng, pid=m["pid"]) for m in mates[: rng.randint(0, 2)]])


# ---------------------------------------------------------------------------
# Secret burners
# ---------------------------------------------------------------------------


def _burner_voices(F: "sfe._Pass", budget: Dict[str, int]) -> None:
    ps = _ps(F.st)
    rng = F.rand("pburner")
    items = list(ps["burners"].items())
    rng.shuffle(items)
    for pid, b in items:
        if b.get("unmasked") or budget["burner"] <= 0:
            continue
        pp = persona(F, pid)
        if not pp or pp["team_id"] not in F.teams:
            continue
        p_post = 0.09 + max(0.0, (50 - pp["morale"]) / 250.0) + (0.15 if pp["banned"] else 0.0) + (0.05 if pp["role_sat"] < 45 else 0.0)
        if rng.random() > p_post:
            continue
        budget["burner"] -= 1
        tm = F.team(pp["team_id"])
        mates = sorted([e for e in _by_team(F).get(pp["team_id"], []) if str(e.get("player_id")) != pid], key=lambda e: -float(e.get("overall") or 0))
        mate_last = sfe._last(str(mates[0].get("player_name"))) if mates else "the new guy"
        topics = ["self", "self", "leak", "fans", "money"]
        if pp["coach_trust"] < 55 or pp["role_sat"] < 55:
            topics += ["coach", "coach"]
        topic = rng.choice(topics)
        ctx = {"last": pp["last"], "nick": tm.get("nick"), "coach_last": sfe._last(tm.get("coach") or "the coach"), "mate_last": mate_last}
        text = _fresh(F, rng, BURNER_LINES[topic], ctx)
        acct = {"id": f"pburner_{sfe._slug(b['handle'])}", "name": b["handle"].lstrip("@").replace("_", " "), "handle": b["handle"], "type": "burner",
                "badge": "anon", "verified": False}
        post = sfe._add_post(F, acct, text, kind="player_burner", cat="players", when=F.stamp(F.iso, 23 * 60, 23 * 60 + 59, rng) if rng.random() < 0.5 else F.stamp(F.iso, 12 * 60, 22 * 60, rng),
                             team_ids=[pp["team_id"]], mag=0.8 + b["suspicion"] / 80.0, controversy=0.6, knowledge="claim", platform="burner")
        if not post:
            continue
        post["burner_suspicion"] = round(b["suspicion"])
        b["posts"] = int(b.get("posts") or 0) + 1
        gain = rng.uniform(5, 11) + (6 if topic == "self" else 0) + (4 if topic == "leak" else 0)
        b["suspicion"] = round(min(100.0, float(b.get("suspicion") or 0) + gain), 1)
        if rng.random() < 0.55:
            ev = sfe._fill(rng.choice(BURNER_EVIDENCE), {"last": pp["last"], "nick": tm.get("nick"), "city": tm.get("city"), "emoji": pp["emoji"],
                                                        "n": max(2, int(b["posts"] * 0.7)), "m": max(3, b["posts"])})
            if ev and ev not in b["evidence"]:
                b["evidence"] = (b["evidence"] + [ev])[-6:]
        _sleuth_update(F, rng, pid, b, pp, tm)


def _sleuth_update(F, rng, pid, b, pp, tm) -> None:
    # Who do the sleuths think it is? Early on they often have the wrong guy.
    if b["suspicion"] >= 30 and (not b.get("accused") or (b.get("accused_id") != pid and rng.random() < b["suspicion"] / 140.0)):
        wrong = [e for e in _by_team(F).get(pp["team_id"], []) if str(e.get("player_id")) != pid]
        if b["suspicion"] < 70 and wrong and rng.random() < 0.4:
            w = rng.choice(wrong)
            b["accused"], b["accused_id"] = str(w.get("player_name")), str(w.get("player_id"))
        else:
            b["accused"], b["accused_id"] = pp["name"], pid
    if b["suspicion"] >= 45 and not b.get("thread"):
        b["thread"] = True
        ev = b["evidence"] or [f"Only ever talks about the {tm.get('nick')}"]
        sfe._add_thread(F, sub=tm.get("sub") or "r/hockey", title=f"[Investigation] Is {b['handle']} secretly {b['accused'] or 'one of our players'}? A thread 🧵",
                        body="Evidence so far:\n" + "\n".join(f"• {e}" for e in ev),
                        kind="burner_investigation", cat="players", flair="Investigation", when=F.stamp(F.iso, 19 * 60, 23 * 60 + 40, rng),
                        comments=[sfe._comment(sfe._reddit_user(rng, tm), "the emoji thing is damning. case closed", rng.randint(200, 1600), sent=-0.2),
                                  sfe._comment(sfe._reddit_user(rng, tm), "you people are insane and I love you", rng.randint(150, 1200), sent=0.3),
                                  sfe._comment(sfe._reddit_user(rng), "FBI wants to know your location OP", rng.randint(90, 900), sent=0.2),
                                  sfe._comment(sfe._reddit_user(rng, tm), f"it's obviously {b['accused'] or 'him'}. nobody else defends {pp['last']} like that", rng.randint(60, 500), sent=-0.1)],
                        team_ids=[pp["team_id"]], mag=1.8, sentiment=0.0, knowledge="speculation")
    unmask = b["suspicion"] >= 100 or (b["suspicion"] >= 78 and rng.random() < 0.18)
    if unmask:
        _unmask(F, rng, pid, b, pp, tm)


def _unmask(F, rng, pid, b, pp, tm) -> None:
    b["unmasked"] = True
    b["unmasked_iso"] = F.iso
    b["accused"], b["accused_id"] = pp["name"], pid
    ins = sfe._insider("lee")
    sfe._add_post(F, ins, f"Confirmed: the anonymous account {b['handle']} — which has spent weeks defending {pp['last']} and criticizing the {tm.get('nick')} — is run by {pp['name']}. "
                  f"The account was deleted within the hour.", kind="burner_unmasked", cat="players", when=F.stamp(F.iso, 12 * 60, 20 * 60, rng),
                  team_ids=[pp["team_id"]], player_id=pid, player_name=pp["name"], mag=2.8, star=F.star({"ovr": pp["ovr"] or 75}), controversy=1.0, knowledge="confirmed")
    sfe._add_post(F, sfe._meme_account(rng), sfe._pick_fresh(F.st, rng, [f"{pp['last']} defending {pp['last']} from {b['handle']} for three months is the funniest thing this sport has ever produced",
                                                                        f"BREAKING: local man {pp['last']} is his own biggest fan",
                                                                        f"{pp['last']} after getting caught running a burner: 'my cousin had my phone'"]),
                  kind="burner_unmasked", cat="players", when=F.stamp(F.iso, 20 * 60, 23 * 60 + 50, rng), team_ids=[pp["team_id"]],
                  player_id=pid, player_name=pp["name"], mag=2.4, controversy=0.8)
    _incident(F, pp, "burner_unmasked", f"Ran the anonymous account {b['handle']}", severity=3, target="team",
              headline=f"{pp['name']} unmasked as anonymous account {b['handle']}",
              summary=f"{pp['name']} ran {b['handle']}, an anonymous account that praised him and took shots at the {tm.get('nick')} coaching staff and teammates.")


# ---------------------------------------------------------------------------
# Event-driven voices (trades, injuries) — safe to run on unplayed days
# ---------------------------------------------------------------------------


def gen_player_event_voices(F: "sfe._Pass") -> None:
    league = getattr(getattr(F.session, "sim", None), "league", None)
    rows, seen = sfe._seen(F.st, "ps_trade")
    for tr in list(getattr(league, "trade_history", None) or [])[-20:]:
        key = str(tr.get("trade_id") or "")
        if not key or key in seen or not tr.get("accepted", True):
            continue
        sfe._mark_seen(F.st, "ps_trade", key)
        if sfe._si(tr.get("calendar_day"), F.day_idx) < F.day_idx - 3:
            continue
        reason = str(tr.get("reason_text") or "").lower()
        rng = F.rand("ptrade", key)
        for m in (tr.get("moved_players") or [])[:3]:
            pid = str(m.get("asset_id") or m.get("player_id") or "")
            pp = persona(F, pid)
            if not pp:
                continue
            new_tid = str(m.get("acquiring_team_id") or "")
            old_tid = str(m.get("from_team_id") or m.get("sending_team_id") or "")
            if not old_tid:
                others = [t for t in (tr.get("participating_teams") or []) if str(t) != new_tid]
                old_tid = str(others[0]) if others else ""
            new, old = F.team(new_tid), F.team(old_tid)
            if not new or pp["arch"] == "lurker" and rng.random() < 0.7:
                continue
            pp = {**pp, "team_id": new_tid or pp["team_id"]}
            if ("demand" in reason or "fractured" in reason or "wish" in reason) and pp["heat"] >= 0.45:
                text = sfe._pick_fresh(F.st, rng, ["free at last 🕊️", f"some places just aren't a fit. new city, new me. let's go {new.get('nick')}",
                                                   f"{new.get('city')}. finally somewhere that wants me 🙏"])
                sent = -0.2
            else:
                text = sfe._pick_fresh(F.st, rng, [f"Thank you {old.get('city') or 'to my old team'}. It was an honour. Excited for the next chapter with the {new.get('nick')}. 🙏",
                                                   f"{old.get('city')} ❤️ forever. {new.get('city')}, let's get to work.",
                                                   f"Packing the car. Grateful for every day in {old.get('city')}. Can't wait to meet the boys in {new.get('city')}."])
                sent = 0.5
            post = _player_post(F, pp, text, kind="player_trade", when=F.stamp(F.iso, 14 * 60, 23 * 60, rng), sentiment=sent, mag=1.5,
                                extra_teams=[old_tid] if old_tid else [])
            if post:
                mates = _teammates(F, new_tid, pid, rng, n=2)
                _add_replies(post, [_reply_row(F, _acct(F, mt), sfe._pick_fresh(F.st, rng, [f"welcome to {new.get('city')} brother 🤝", "let's gooo 🔥", "locker next to mine. no pressure 😂", "first dinner's on you"]),
                                               rng, pid=mt["pid"]) for mt in mates])
                if sent < 0:
                    sfe._add_post(F, sfe._fan_account(F, old_tid, "doomer", rng), sfe._pick_fresh(F.st, rng, [f"'{text[:30]}' after everything this city did for him? bye 👋",
                                  "don't let the door hit you on the way out", "the PR team did not approve that post lmao", "we'll see how free he feels in February",
                                  "wow. ok. we're booing him when he comes back right?"]),
                                  kind="trade_react", cat="players", when=F.stamp(F.iso, 15 * 60, 23 * 60 + 30, rng), team_ids=[old_tid], mag=0.9,
                                  sentiment=-0.7, controversy=0.6)
    for pid, ent in list(_entities(F.session).items()):
        soc = ent.get("social") if isinstance(ent, dict) else None
        q = soc.pop("queued_post", None) if isinstance(soc, dict) else None
        if not q:
            continue
        pp = persona(F, pid)
        if pp:
            rng = F.rand("pqueued", pid)
            _player_post(F, pp, str(q.get("text") or ""), kind=str(q.get("kind") or "player_post"), when=F.stamp(F.iso, 10 * 60, 18 * 60, rng),
                         sentiment=0.2, controversy=0.2, mag=1.3)
    rows, seen = sfe._seen(F.st, "ps_inj")
    for row in list(getattr(F.session, "injury_log_all", None) or [])[-40:]:
        key = str(row.get("id") or "")
        if not key or key in seen or str(row.get("calendar_iso") or "")[:10] > F.iso:
            continue
        sfe._mark_seen(F.st, "ps_inj", key)
        games = sfe._si(row.get("games_initial") or row.get("games"))
        if games < 8 or sfe._iso_days_between(str(row.get("calendar_iso") or F.iso), F.iso) > 3:
            continue
        pp = persona(F, str(row.get("player_id") or ""))
        if not pp or pp["arch"] in ("lurker",):
            continue
        rng = F.rand("pinj", key)
        text = sfe._pick_fresh(F.st, rng, ["Day 1 of the comeback. 💪", "Frustrating, but I'll be back stronger. Thanks for all the messages 🙏",
                                           "not how I wanted the season to go. see you soon.", "🙏", f"{pp['dog']} is my rehab coach now. strict guy 🐶"])
        _player_post(F, pp, text, kind="player_injury", when=F.stamp(F.iso, 12 * 60, 21 * 60, rng), sentiment=0.1, mag=1.1)


# ---------------------------------------------------------------------------
# GM burner interplay + API summary
# ---------------------------------------------------------------------------


def react_to_gm_burner(session: Any, text: str, tone: str) -> None:
    """A volatile player named in a negative post from the GM's burner fires back."""
    if tone not in ("criticism", "trade_talk"):
        return
    try:
        with sfe._LOCK:
            day_idx = sfe._current_day(session)
            iso = sfe._iso_for_day(session, day_idx)
            if not iso:
                return
            F = sfe._Pass(session, iso=iso, day_idx=day_idx, mode="external")
            low = str(text or "").lower()
            for e in _by_team(F).get(F.utid, []):
                last = sfe._last(str(e.get("player_name") or "")).lower()
                if len(last) < 4 or last not in low:
                    continue
                pp = persona(F, str(e.get("player_id")))
                if not pp or pp["heat"] < 0.4:
                    continue
                rng = F.rand("gmburn", text[:30])
                acct = (getattr(session, "gm_burner_account", None) or {}).get("handle") or "that anonymous account"
                reply = sfe._pick_fresh(F.st, rng, [f"whoever runs {acct}: say it to my face 👋", f"funny how {acct} always knows what's said in our room 🤔",
                                                    f"{acct} has 40 followers and an opinion on my game. cute."])
                _player_post(F, pp, reply, kind="player_clapback", when=F.stamp(iso, 9 * 60, 23 * 60, rng), sentiment=-0.6, controversy=0.8, mag=1.6)
                acct_row = getattr(session, "gm_burner_account", None)
                if isinstance(acct_row, dict):
                    acct_row["suspicion_score"] = min(100.0, float(acct_row.get("suspicion_score") or 0) + 6.0)
                break
            if F.posts:
                session.social_posts = list(getattr(session, "social_posts", None) or []) + F.posts
    except Exception:
        _log.debug("gm burner clapback failed", exc_info=True)


def player_social_summary(session: Any) -> Dict[str, Any]:
    st = sfe._state(session)
    ps = _ps(st)
    utid = str(getattr(session, "user_team_id", "") or "")
    try:
        from app.sim_engine.franchise.storyline_engine import _player_index_by_id  # noqa: WPS433

        pidx = _player_index_by_id(session) or {}
    except Exception:
        pidx = {}
    ents = _entities(session)

    def av(pid: str) -> Dict[str, Any]:
        p = pidx.get(str(pid))
        return {**(sfe._headshot_bits(p) if p is not None else {}), "player_id": str(pid)}

    watch = []
    for pid, b in ps["burners"].items():
        if b.get("suspicion", 0) < 20 and not b.get("unmasked"):
            continue
        ent = ents.get(pid) or {}
        team_id = str(ent.get("team_id") or "")
        acc_id = str(b.get("accused_id") or "")
        watch.append({
            "handle": b.get("handle"), "suspicion": round(float(b.get("suspicion") or 0)), "posts": int(b.get("posts") or 0),
            "evidence": list(b.get("evidence") or [])[-4:], "team_id": team_id, "is_user_team": team_id == utid,
            "accused_name": b.get("accused") or "", "accused_id": acc_id, "accused_avatar": av(acc_id) if acc_id else {},
            "unmasked": bool(b.get("unmasked")), "owner_name": ent.get("player_name") if b.get("unmasked") else "",
            "unmasked_iso": b.get("unmasked_iso") or "",
        })
    watch.sort(key=lambda r: (not r["is_user_team"], r["unmasked"], -r["suspicion"]))
    accounts = []
    for pid, ent in ents.items():
        if str(ent.get("team_id") or "") != utid or not bool(ent.get("active_roster", True)):
            continue
        base = ps["personas"].get(pid) or {}
        soc = ent.get("social") or {}
        state = ent.get("state") or {}
        accounts.append({
            "player_id": pid, "name": ent.get("player_name"), "handle": base.get("handle") or soc.get("handle"),
            "persona": PERSONA_LABELS.get(base.get("arch"), ""), "chaos": round(float(base.get("chaos") or 0), 2),
            "followers": int(soc.get("followers") or 0), "fan_sentiment": round(float(soc.get("fan_sentiment", 55) or 55)),
            "morale": round(float(state.get("morale", 60) or 60)), "media_stress": round(float(state.get("media_stress", 40) or 40)),
            "posting_ban": int(soc.get("posting_ban_until", -1) or -1) >= sfe._current_day(session),
            "incident": soc.get("incident") if isinstance(soc.get("incident"), dict) else None,
            "avatar": av(pid), "position": ent.get("position"),
        })
    accounts.sort(key=lambda r: -r["followers"])
    incidents = list(reversed(ps["incidents"][-14:]))
    for row in incidents:
        row.setdefault("avatar", av(row.get("player_id")))
    return {"burner_watch": watch[:12], "accounts": accounts[:26], "incidents": incidents}


# ---------------------------------------------------------------------------
# Meetings hook
# ---------------------------------------------------------------------------


def apply_social_meeting_choice(session: Any, player_id: str, choice_id: str) -> None:
    """Side effects of the 'address his social media' meeting choices."""
    ent = _entities(session).get(str(player_id))
    if not isinstance(ent, dict):
        return
    social = ent.setdefault("social", {})
    day = sfe._current_day(session)
    inc = social.get("incident") if isinstance(social.get("incident"), dict) else {}
    if choice_id == "take_it_down":
        social["fan_sentiment"] = round(_clip(float(social.get("fan_sentiment", 55) or 55) + 4), 1)
        inc["handled"] = "apology"
        social["queued_post"] = {"kind": "player_apology", "text": random.Random(f"{player_id}{day}").choice([
            "I want to apologize for my post. Emotions were high and I said things I shouldn't have. That's not who I am and it's not what this team is about.",
            "Took that post down. I was frustrated and I handled it the wrong way. Talked to the guys. We move forward together.",
            "Bad post, bad timing, my bad. Focus is on the next game. 🙏"])}
    elif choice_id == "back_him":
        social["queued_post"] = {"kind": "player_post", "text": "Appreciate the support from the top. Nothing changes. Let's go. 🏒"}
    elif choice_id == "posting_ban":
        social["posting_ban_until"] = day + 21
        inc["handled"] = "ban"
    if choice_id == "back_him":
        social["fan_sentiment"] = round(_clip(float(social.get("fan_sentiment", 55) or 55) + 2), 1)
        inc["handled"] = "backed"
    elif choice_id == "laugh_off":
        inc["handled"] = "laughed"
    if inc:
        inc["handled_day"] = day
        social["incident"] = inc
