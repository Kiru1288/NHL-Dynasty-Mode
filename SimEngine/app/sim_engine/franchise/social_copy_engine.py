"""Compositional social copy generation for Puckr / IceHole feeds."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reporter fragments — opener / clause / closer per narrative_angle
# ---------------------------------------------------------------------------

def _frag(openers: List[str], clauses: List[str], closers: List[str]) -> Dict[str, List[str]]:
    return {"openers": openers, "clauses": clauses, "closers": closers}


REPORTER_FRAGMENTS: Dict[str, Dict[str, List[str]]] = {
    "trade_market": _frag(
        [
            "Sources around {team} say {name} remains in active trade conversations",
            "League execs continue to monitor {name}'s availability with {team}",
            "The market on {name} has not cooled — {points} points on {games_played} GP keeps interest alive",
            "Multiple clubs have checked in on {name} ({overall} OVR) per league contacts",
            "",
            "Trade chatter on {name} is louder than {team}'s {team_record} record might suggest",
            "Hockey ops circles list {name} among movable pieces if {team} pivots",
            "Return ask on {name} starts with a first-round pick plus a prospect, per one rival exec",
            "Two teams have asked about {name}'s contract ({cap_hit}M) this week alone",
            "Western conference GM mentions {name} when asked about rental prices",
            "Eastern desk hears {team} is listening on {name} at {ppg} PPG",
            "Agent market for {name} is active despite public silence from {team}",
            "Cap-strapped clubs eye {name} as a partial salary solution",
            "Playoff teams view {name} as a depth upgrade if price drops",
            "Rebuilders ask whether {name} is truly off limits in {team}",
        ],
        [
            " with return packages still being shaped",
            " and cap mechanics ({cap_hit}M) complicating the math",
            " despite public denials from the club",
            " as {ppg} PPG production holds the ask firm",
        ],
        [
            " Nothing imminent, but the phone lines stay open.",
            " A decision may come before the deadline pressure peaks.",
            " Both sides are measuring leverage carefully.",
            "",
        ],
    ),
    "slump_watch": _frag(
        [
            "{name} is on a cold stretch — {points} points in {games_played} games for {team}",
            "Production watch: {name} at {ppg} PPG vs {expected_points} expected through {games_played} GP",
            "{team} needs more from {name} ({overall} OVR) during this {team_record} run",
            "",
            "The numbers on {name} tell a rough story: {goals}G-{assists}A in {games_played} appearances",
        ],
        [
            " and the coaching staff is rotating usage",
            " with linemates shuffled twice in ten days",
            " while media pressure builds in a {team_record} context",
            " as confidence metrics slide league-wide",
        ],
        [
            " Internal patience exists — but not indefinitely.",
            " The next five games likely decide the narrative.",
            "",
            " Slump or adjustment — the room will know soon.",
        ],
    ),
    "goaltending": _frag(
        [
            "{team}'s crease story centers on {save_pct} save percentage across {games_played} starts",
            "Goaltending note: {gaa} GAA through {games_played} appearances for {team}",
            "The net has been unstable — {save_pct} SV% vs {expected_save_pct} expected",
            "",
            "Between the pipes, {team} is living at {save_pct} with {gaa} GAA",
        ],
        [
            " and starter usage is under review",
            " with backup options getting live reps",
            " as {team_record} tightens the margin for error",
            " in a market that punishes crease volatility",
        ],
        [
            " A starter decision could land within the week.",
            "",
            " The numbers usually force the coaching staff's hand.",
            " Playoff math makes this unavoidable.",
        ],
    ),
    "draft_board": _frag(
        [
            "Draft desk: {name} trending on boards after {points} points in {games_played} junior games",
            "Scouts have {name} ({overall} OVR projection) climbing post-{team_record} tournament run",
            "The {team} pipeline conversation includes {name} at {ppg} PPG pace",
            "",
            "Combine buzz on {name} — {goals} goals in {games_played} GP turning heads",
        ],
        [
            " with lottery teams doing homework",
            " and ranking services moving him up boards",
            " as {age}-year-old timelines compress",
            " amid a thin positional class",
        ],
        [
            " Mock drafts will chase this for months.",
            "",
            " Real picks depend on lottery night.",
            " The interview circuit starts soon.",
        ],
    ),
    "contract_battle": _frag(
        [
            "Contract file: {name} and {team} navigating {cap_hit}M AAV against {points}-point production",
            "{name}'s next deal hinges on {ppg} PPG and a {team_record} team context",
            "Cap math for {team}: {name} at {cap_hit}M with {games_played} GP logged",
            "",
            "Extension talks on {name} ({overall} OVR) remain unresolved",
        ],
        [
            " with arb comparables tightening",
            " and term length the sticking point",
            " as UFA/CBA clocks tick",
            " while agent leverage holds",
        ],
        [
            " Both sides want a number — neither wants to blink first.",
            "",
            " Deadline pressure may force clarity.",
            " The cap sheet leaves limited wiggle room.",
        ],
    ),
    "conduct_desk": _frag(
        [
            "League sources confirm an active review involving {name} and {team}",
            "Conduct desk: {name} situation under league/office scrutiny",
            "Investigation track — {team} cooperating as {name}'s availability status evolves",
            "",
            "{name} ({overall} OVR) linked to off-ice review; {team} issued no detailed comment",
        ],
        [
            " with legal counsel engaged on both sides",
            " and timeline uncertain",
            " as public knowledge remains limited",
            " pending official findings",
        ],
        [
            " More when the league permits disclosure.",
            "",
            " This file is open — updates expected.",
            " No discipline announced yet.",
        ],
    ),
    "injury_watch": _frag(
        [
            "{name} listed with injury concern — {games_played} GP season paused at {points} points",
            "Medical update: {name} ({overall} OVR) day-to-day; {team} at {team_record}",
            "Injury wire: {name} — {injury_type} tag, timeline TBD",
            "",
            "{team} without full services of {name} after latest medical review",
        ],
        [
            " and lineup combinations shifting",
            " with IR eligibility under evaluation",
            " as {team} manages minutes",
            " in a compressed schedule stretch",
        ],
        [
            " Fantasy and roster managers should monitor daily.",
            "",
            " Return-to-play clearance is the next milestone.",
            " The club will update when skating resumes.",
        ],
    ),
    "locker_room": _frag(
        [
            "Locker room pulse: {team} navigating internal tension around {name}",
            "Teammates describe a charged room after latest {team_record} result",
            "{name}'s role and usage fueling whispers inside {team}'s dressing room",
            "",
            "Culture check — {team} leaders addressing group dynamics post-{points}-point week from {name}",
        ],
        [
            " with veterans pushing for direct conversation",
            " and coaching staff mediating minutes",
            " as media heat rises locally",
            " while outside noise grows",
        ],
        [
            " Winning usually quiets this — losing amplifies it.",
            "",
            " The next homestand matters for morale.",
            " Leadership meetings continue behind closed doors.",
        ],
    ),
    "league_wire": _frag(
        [
            "League wire: {team} at {team_record}, rank {league_rank} — {name} with {points} points",
            "{name} ({overall} OVR) remains a nightly factor for {team} at {ppg} PPG",
            "Around the league: {team}'s {team_record} mark tied to {name}'s {goals}G-{assists}A line",
            "",
            "Notebook: {name} through {games_played} GP — production meets {expected_points} expected pace",
        ],
        [
            " in a crowded standings picture",
            " with playoff math tightening",
            " as trade deadline season approaches",
            " and special teams usage shifting",
        ],
        [
            " More league-wide notes to follow.",
            "",
            " Standard regular-season noise — for now.",
            " Context changes weekly this time of year.",
        ],
    ),
}

# Player mood skeletons (before voice filter)
MOOD_SKELETONS: Dict[str, Dict[str, List[str]]] = {
    "win": _frag(
        ["Good win for {team} — {points} points from me on {games_played} GP", "Two points. Room felt it.", "Team effort tonight. On to the next one."],
        [" We stayed disciplined.", " The crowd carried us.", " Special teams showed up."],
        [" Back at it tomorrow.", "", " Momentum matters now."],
    ),
    "loss": _frag(
        ["Not our night — {team} at {team_record} hurts", "I own my piece of this.", "{ppg} PPG isn't enough when we lose like that."],
        [" No excuses in our room.", " We get one chance to respond.", " Details beat us."],
        [" Learn and move.", "", " Standard isn't optional."],
    ),
    "frustrated": _frag(
        ["Funny how {points} points becomes the whole story", "Everyone has opinions on {name} — I have the workload", "The narrative around {team} gets loud fast"],
        [" I said what I said.", " Focus stays internal.", " Noise is noise."],
        [" Moving on.", "", " Next shift matters."],
    ),
    "grateful": _frag(
        ["Grateful for {team} and this city", "Community day reminded me why we play", "Fans showed up — we felt it"],
        [" Honored to represent {team}.", " This job is bigger than stats.", " Thankful for the room."],
        [" Back to work.", "", " Appreciate the support."],
    ),
    "hype": _frag(
        ["Big game coming — {team} at {team_record} with everything on the line", "Playoff push mode: {points} points and counting", "Let's go — {name} ready for the moment"],
        [" Arena will be rocking.", " This is why you play.", " Energy is high."],
        [" See you at puck drop.", "", " LFG."],
    ),
}

# Ambient Puckr fan lines (~40 per sentiment, abbreviated sets expanded in code)
AMBIENT_FAN: Dict[str, List[str]] = {
    "hype": [
        "{team} is cooking — {ppg} PPG from {name} and the room looks alive",
        "If {name} keeps this {points}-point pace, {team} is a problem",
        "Love this {team_record} energy. Stay humble though.",
        "Chart crime: {name} at {overall} OVR playing like a star",
        "Crowd was unreal. {team} belongs in this conversation",
        "Special teams + {name} = win formula. Book it.",
        "This is the version of {team} we waited for",
        "Playoff hockey already in {team}'s building",
        "{name} with {goals} goals — elite company",
        "Front office earned some trust tonight",
    ] * 4,
    "outrage": [
        "{team} at {team_record} is unacceptable — {name} can't carry this alone",
        "Fire someone. {ppg} PPG from {name} and we still lose",
        "Cap hit {cap_hit}M for this? Brutal.",
        "Another night, another collapse. {team} looks lost",
        "Trade {name}? At this point explore everything",
        "Coaching staff is out of answers — {points} points mean nothing in losses",
        "Embarrassing home effort. {team_record} says it all",
        "Media soft on {team}. This roster has holes",
        "Goalie at {save_pct} SV% — pick a lane",
        "Owner should be furious watching {team_record}",
    ] * 4,
    "meme": [
        "POV: you're {name} checking Puckr after a {team_record} week",
        "{team} hockey: it is what it is (pain)",
        "Me explaining why {ppg} PPG is actually good actually",
        "Caps fan voice: {cap_hit}M is totally fine actually",
        "Hot take machine says trade {name} for a bag of pucks",
        "Advanced stats say {name} is fine. Eye test says help",
        "Narrator: {team} did not, in fact, bounce back",
        "This team is a group project and {name} did the whole thing",
        "Imagine supporting {team} in this economy",
        "I am not saying fire the coach but… {team_record}",
    ] * 4,
    "concern": [
        "Quietly worried about {name} — production at {points} through {games_played} GP",
        "{team}'s {team_record} trend is not a blip",
        "Injury cloud over {name} — watch the next skate",
        "Depth scoring vanished. {name} can't do everything",
        "Playoff math getting tight at {league_rank}",
        "Save percentage {save_pct} — crease is a story",
        "Contract year for {name} adds pressure to {ppg} PPG",
        "Room vibes feel off despite {points} points from stars",
        "Trade rumors around {name} might be a distraction",
        "Long road trip could break or make {team}",
    ] * 4,
}

_REPORTER_STYLE_OVERRIDES: Dict[str, Any] = {
    "hart": lambda t: t.replace("Sources", "HOT TAKE: Sources").replace(".", " — and I'm not sorry."),
    "reid": lambda t: f"{t} (cap table attached)" if "cap" not in t.lower() else t,
    "morin": lambda t: t.replace("Internal", "Internally").replace("pressure", "pressure — but the room is still fighting"),
    "lee": lambda t: re.sub(r"\b(say|says|said)\b", "allege", t, count=1) if "allege" not in t else t,
    "ellison": lambda t: t if t.startswith("BREAKING") else f"BREAKING: {t}",
}

_VOICE_FILTERS: Dict[str, Any] = {
    "quiet": lambda t: t.split(".")[0] + "." if "." in t else t[:80],
    "polished": lambda t: t.replace("LFG", "Looking forward").replace("…", "."),
    "team_first": lambda t: f"We — {t.lower()}" if not t.lower().startswith("we") else t,
    "playful": lambda t: t + " 😅" if len(t) < 200 else t,
    "online": lambda t: t.upper() if len(t) < 60 else t,
}


def _safe_format(template: str, ctx: Dict[str, Any]) -> str:
    if not template:
        return ""
    try:
        return template.format(**ctx)
    except (KeyError, ValueError, IndexError):
        return ""


def _stat_int(row: Dict[str, Any], *keys: str) -> int:
    for key in keys:
        val = row.get(key)
        if val is None or val == "":
            continue
        try:
            return int(float(val))
        except (TypeError, ValueError):
            continue
    return 0


def _team_record_label(session: Any, team_id: str) -> str:
    if not session or not team_id:
        return "—"
    standings = getattr(session, "standings", None) or getattr(session, "league_standings", None) or []
    for row in standings:
        if str(row.get("team_id") or row.get("id") or "") != str(team_id):
            continue
        w = _stat_int(row, "w", "wins")
        l = _stat_int(row, "l", "losses")
        otl = _stat_int(row, "otl", "ot_losses", "ot")
        if w or l or otl:
            return f"{w}-{l}-{otl}"
    team = (getattr(session, "team_by_id", None) or {}).get(str(team_id))
    if team is not None:
        w = int(getattr(team, "wins", 0) or 0)
        l = int(getattr(team, "losses", 0) or 0)
        otl = int(getattr(team, "otl", 0) or getattr(team, "ot_losses", 0) or 0)
        if w or l or otl:
            return f"{w}-{l}-{otl}"
    return "—"


def _lookup_session_evidence(session: Any, storyline: Dict[str, Any]) -> Dict[str, Any]:
    if session is None:
        return {}
    pid = str(storyline.get("player_id") or "")
    tid = str(storyline.get("team_id") or "")
    out: Dict[str, Any] = {}
    stats = dict(getattr(session, "player_season_stats", None) or {})
    row = stats.get(pid) if pid else None
    if isinstance(row, dict):
        gp = _stat_int(row, "gp", "games_played")
        goals = _stat_int(row, "g", "goals")
        assists = _stat_int(row, "a", "assists")
        points = _stat_int(row, "pts", "points")
        if points <= 0 and (goals or assists):
            points = goals + assists
        out["games_played"] = gp
        out["goals"] = goals
        out["assists"] = assists
        out["points"] = points
        if gp > 0:
            out["ppg"] = round(points / gp, 2)
        sv = row.get("save_pct") or row.get("sv_pct")
        if sv not in (None, ""):
            out["save_pct"] = round(float(sv), 3) if float(sv) <= 1 else round(float(sv) / 100, 3)
        gaa = row.get("gaa")
        if gaa not in (None, ""):
            out["gaa"] = round(float(gaa), 2)
        if not tid:
            tid = str(row.get("team_id") or "")
        pname = str(row.get("name") or "").strip()
        if pname:
            out["name"] = pname
        ovr = row.get("overall") or row.get("ovr")
        if ovr not in (None, "", 0):
            out["overall"] = round(float(ovr), 1)

    if pid:
        for tm in (getattr(session, "team_by_id", None) or {}).values():
            for player in getattr(tm, "roster", None) or []:
                if str(getattr(player, "id", "") or "") != pid:
                    continue
                if "name" not in out:
                    out["name"] = str(getattr(player, "name", "") or getattr(player, "player_name", "") or "").strip()
                if "overall" not in out or not out.get("overall"):
                    ovr = getattr(player, "overall", None) or getattr(player, "ovr", None)
                    if ovr not in (None, "", 0):
                        out["overall"] = round(float(ovr), 1)
                cap = getattr(player, "cap_hit", None) or getattr(player, "salary", None)
                if cap not in (None, "", 0):
                    cap_m = float(cap)
                    if cap_m > 1000:
                        cap_m /= 1_000_000
                    out["cap_hit"] = round(cap_m, 2)
                if not tid:
                    tid = str(getattr(tm, "id", "") or "")
                break

    if tid and "team_record" not in out:
        out["team_record"] = _team_record_label(session, tid)
    team_name = str(storyline.get("team_name") or "").strip()
    if team_name:
        out["team"] = team_name
    elif tid:
        tm = (getattr(session, "team_by_id", None) or {}).get(str(tid))
        if tm is not None:
            out["team"] = str(getattr(tm, "name", "") or getattr(tm, "city", "") or tid)
    return {k: v for k, v in out.items() if v not in (None, "")}


_BROKEN_SOCIAL_PATTERNS = (
    re.compile(r"\bthe player\b", re.I),
    re.compile(r"\(\s*0\s*ovr\s*\)", re.I),
    re.compile(r"\b0 points in 0 games\b", re.I),
    re.compile(r"\bthrough 0 gp\b", re.I),
    re.compile(r"\b0 starts\b", re.I),
    re.compile(r"\b0\.00 ppg through 0\b", re.I),
    re.compile(r"\{[a-z_]+\}"),
)


def _looks_like_broken_social_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if len(cleaned) < 8:
        return True
    return any(p.search(cleaned) for p in _BROKEN_SOCIAL_PATTERNS)


def _headline_fallback_post(storyline: Dict[str, Any], reporter: Dict[str, Any]) -> str:
    headline = str(storyline.get("headline") or "").strip()
    summary = str(storyline.get("summary") or "").strip()
    body = headline or summary
    if not body:
        team = str(storyline.get("team_name") or "the club")
        pname = str(storyline.get("player_name") or "").strip()
        body = f"{pname} remains a storyline around {team}." if pname else f"League desk tracking a developing story around {team}."
    outlet = str(reporter.get("outlet") or "Desk").strip()
    return _apply_reporter_voice(f"{outlet} — {body}"[:280], reporter)


def build_evidence_context(storyline: Dict[str, Any], session: Any = None) -> Dict[str, Any]:
    ev = dict(storyline.get("evidence") or {})
    if session is not None:
        enriched = _lookup_session_evidence(session, storyline)
        for key, val in enriched.items():
            if key not in ev or ev.get(key) in (None, "", 0, "0", "0.00", ".900", "—"):
                ev[key] = val
    pname = str(storyline.get("player_name") or ev.get("name") or "").strip()
    if not pname or pname.lower() == "the player":
        pname = str(ev.get("name") or "").strip()
    ctx = {
        "name": pname or "Unknown player",
        "team": str(storyline.get("team_name") or ev.get("team") or "the club"),
        "ppg": ev.get("ppg", ev.get("points_per_game", storyline.get("ppg", "0.00"))),
        "points": ev.get("points", storyline.get("points", 0)),
        "goals": ev.get("goals", 0),
        "assists": ev.get("assists", 0),
        "games_played": ev.get("games_played", ev.get("last_n", 0)),
        "overall": ev.get("overall", storyline.get("player_overall", 0)),
        "cap_hit": ev.get("cap_hit", 0),
        "team_record": ev.get("team_record", "—"),
        "league_rank": ev.get("league_rank", ev.get("rank", "—")),
        "save_pct": ev.get("save_pct", ".900"),
        "gaa": ev.get("gaa", "2.80"),
        "expected_save_pct": ev.get("expected_save_pct", ".905"),
        "expected_points": ev.get("expected_points", ev.get("points", 0)),
        "injury_type": ev.get("injury_type", "undisclosed"),
        "age": ev.get("age", storyline.get("age", 0)),
        "heat": int(storyline.get("heat") or 0),
    }
    for k, v in list(ctx.items()):
        if isinstance(v, float):
            ctx[k] = round(v, 3) if k in ("ppg", "save_pct", "expected_save_pct", "gaa") else round(v, 2)
    return ctx


def build_entity_context(entity: Dict[str, Any]) -> Dict[str, Any]:
    ident = dict(entity.get("identity") or {})
    return {
        "name": str(entity.get("player_name") or ident.get("name") or "Player"),
        "team": str(entity.get("team_name") or entity.get("team_id") or "the team"),
        "points": entity.get("points", entity.get("season_points", 0)),
        "games_played": entity.get("games_played", 0),
        "ppg": entity.get("ppg", 0),
        "overall": entity.get("overall", ident.get("overall", 0)),
        "team_record": entity.get("team_record", "—"),
        "goals": entity.get("goals", 0),
    }


def _active_storyline_boost(session: Any, storyline: Dict[str, Any]) -> float:
    active = list(getattr(session, "active_cause_storylines", None) or [])
    if not active:
        return 1.0
    pid = str(storyline.get("player_id") or "")
    tid = str(storyline.get("team_id") or "")
    for row in active:
        if pid and str(row.get("player_id") or "") == pid:
            return 1.45
        if tid and str(row.get("team_id") or "") == tid:
            return 1.25
    return 1.0


def _pick_slots(frags: Dict[str, List[str]], rng: random.Random, urgent: bool) -> Tuple[str, str, str]:
    weights_o = [2.0 if urgent and f and "?" not in f else 1.0 for f in frags["openers"]]
    weights_c = [1.8 if urgent and f else 1.0 for f in frags["clauses"]]
    weights_e = [1.0 for _ in frags["closers"]]
    opener = rng.choices(frags["openers"], weights=weights_o, k=1)[0]
    clause = rng.choices(frags["clauses"], weights=weights_c, k=1)[0]
    closer = rng.choice(frags["closers"])
    return opener, clause, closer


def _apply_reporter_voice(text: str, reporter: Dict[str, Any]) -> str:
    rid = str(reporter.get("id") or "")
    fn = _REPORTER_STYLE_OVERRIDES.get(rid)
    return fn(text) if fn else text


def voice_filter(text: str, style: str) -> str:
    fn = _VOICE_FILTERS.get(str(style or "polished"))
    return fn(text.strip()) if fn else text.strip()


REPORTER_TAGLINES: List[str] = [
    "Watch the next skate.",
    "File stays open.",
    "Club meeting scheduled.",
    "No trade call confirmed.",
    "Agent circle quiet for now.",
    "Cap sheet still matters here.",
    "Room sentiment mixed.",
    "Practice report due tomorrow.",
    "National desk monitoring.",
    "Beat follow-up expected.",
    "Timeline remains fluid.",
    "Second source pending.",
    "Market price unsettled.",
    "Usage trend worth tracking.",
    "Special teams angle noted.",
    "Playoff math in background.",
    "Health update pending.",
    "Coach comment likely soon.",
    "Front office on record soon.",
    "League office looped in.",
    "Travel schedule complicates.",
    "Home crowd factor real.",
    "Road trip context matters.",
    "Division rival watching.",
    "Contract clock ticking.",
    "Offer sheet chatter faint.",
    "Waiver wire irrelevant.",
    "Farm call-up possible.",
    "Leadership group involved.",
    "Veteran voice expected.",
    "Rookie minutes in question.",
    "Analytics staff split.",
    "Video session planned.",
    "Media day on calendar.",
    "Fan forum noise loud.",
    "Ticket market steady.",
    "Merch spike unlikely.",
    "Broadcast crew circling.",
    "Podcast cycle incoming.",
    "Radio host pushing.",
    "Print deadline tonight.",
    "Radio host pushing.",
    "Notebook filler until skate.",
    "Short shift in coverage.",
    "Late wire crossing.",
    "Early skate to settle.",
    "Cross-border interest minimal.",
    "Scout video loading.",
    "Trade board refresh pending.",
    "Clause language relevant.",
    "Western road swing ahead.",
]


REPORTER_LEADS: List[str] = [
    "Quick hit:", "Desk note:", "Latest:", "Tracking:", "Scene:", "File:", "Wire:", "Beat:",
    "Morning note:", "Post-practice:", "Travel day:", "Road note:", "Home stand:", "Division note:",
    "Cap angle:", "Usage note:", "Health watch:", "Room read:", "Market pulse:", "Trade desk:",
    "Playoff lens:", "Special teams:", "Goalie note:", "Blue line:", "Forward line:",
    "Bottom six:", "Top line:", "Power play:", "Penalty kill:", "Even strength:",
    "Overtime note:", "Shootout note:", "Back-to-back:", "Schedule note:", "Rest advantage:",
    "Rookie watch:", "Veteran lead:", "Captaincy:", "Leadership:", "Culture note:",
    "Analytics:", "Eye test:", "Video review:", "Scout view:", "Fan pulse:",
    "Ownership lens:", "Front office:", "Coaching note:", "Development:", "Farm report:",
]


def compose_reporter_post(
    storyline: Dict[str, Any],
    reporter: Dict[str, Any],
    rng: random.Random,
    session: Any = None,
) -> str:
    angle = str(storyline.get("narrative_angle") or "league_wire")
    frags = REPORTER_FRAGMENTS.get(angle) or REPORTER_FRAGMENTS["league_wire"]
    ctx = build_evidence_context(storyline, session)
    if rng.random() < 0.18:
        parts_name = str(ctx.get("name") or "").split()
        if len(parts_name) >= 2:
            ctx = dict(ctx)
            ctx["name"] = f"{parts_name[-1]}, {parts_name[0]}"
    urgent = _active_storyline_boost(session, storyline) > 1.2 if session else int(ctx.get("heat") or 0) >= 55
    opener, clause, closer = _pick_slots(frags, rng, urgent)
    mode = rng.random()
    if mode < 0.28:
        parts = [_safe_format(opener, ctx)]
    elif mode < 0.52:
        parts = [_safe_format(opener, ctx), _safe_format(clause, ctx)]
    else:
        parts = [_safe_format(opener, ctx), _safe_format(clause, ctx), _safe_format(closer, ctx)]
    parts = [p for p in parts if p]
    if parts and parts[0] and rng.random() < 0.22:
        parts[0] = f"{rng.choice(REPORTER_LEADS)} {parts[0]}"
    stat_keys = ["points", "ppg", "goals", "games_played", "cap_hit", "save_pct", "team_record", "league_rank"]
    stat_key = rng.choice(stat_keys)
    stat_val = ctx.get(stat_key, ctx.get("points", 0))
    if not re.search(r"\d", " ".join(parts)) and stat_val not in (None, "", 0, "—"):
        parts.append(f"({stat_key.replace('_', ' ')}: {stat_val})")
    if rng.random() < 0.22:
        parts.insert(0, f"{reporter.get('outlet', 'Desk')} —")
    if int(ctx.get("heat") or 0) >= 70 and rng.random() < 0.4:
        parts.append(f"Heat index {ctx.get('heat')}.")
    tag = rng.choice(REPORTER_TAGLINES)
    if rng.random() < 0.85:
        parts.append(_safe_format(tag, ctx) if "{" in tag else tag)
    if rng.random() < 0.25:
        parts.append(rng.choice(REPORTER_TAGLINES))
    text = " ".join(parts).strip()
    suffix = rng.choice(["", ".", " — more coming.", " per league sources.", ""])
    if suffix and not text.endswith((".", "!", "?")):
        text = text + suffix
    if _looks_like_broken_social_text(text):
        text = _headline_fallback_post(storyline, reporter)
    return _apply_reporter_voice(text[:280], reporter)


def compose_player_post(entity: Dict[str, Any], mood: str, rng: random.Random) -> str:
    skel = MOOD_SKELETONS.get(mood) or MOOD_SKELETONS["win"]
    ctx = build_entity_context(entity)
    opener, clause, closer = _pick_slots(skel, rng, urgent=False)
    style = str((entity.get("social") or {}).get("style") or "polished")
    raw = " ".join(p for p in (_safe_format(opener, ctx), _safe_format(clause, ctx), _safe_format(closer, ctx)) if p)
    if not re.search(r"\d", raw) and int(ctx.get("points") or 0) > 0:
        raw = f"{raw} ({ctx['points']} pts)".strip()
    return voice_filter(raw[:240], style)


def compose_ambient_fan_post(
    sentiment: str,
    ctx: Dict[str, Any],
    rng: random.Random,
    storyline: Optional[Dict[str, Any]] = None,
    reporter: Optional[Dict[str, Any]] = None,
) -> str:
    pool = AMBIENT_FAN.get(sentiment) or AMBIENT_FAN["concern"]
    for _ in range(6):
        line = rng.choice(pool)
        text = _safe_format(line, ctx).strip()
        if text and not _looks_like_broken_social_text(text):
            if not re.search(r"\d", text) and int(ctx.get("points") or 0) > 0:
                text = f"{text} ({ctx['points']} pts)"
            return text[:260]
    if storyline and reporter:
        return _headline_fallback_post(storyline, reporter)[:260]
    headline = str((storyline or {}).get("headline") or "").strip()
    if headline:
        return headline[:260]
    name = str(ctx.get("name") or "This player")
    team = str(ctx.get("team") or "the club")
    return f"{name} is all over the {team} conversation right now."[:260]
