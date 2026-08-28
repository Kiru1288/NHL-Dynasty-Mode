"""Awards Night fan reaction feed — profiles, tweets, and API route."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException

RANDOM_FAN_API_URL = "https://randomuser.me/api/"

FALLBACK_FAN_FIRST_NAMES = [
    "Mason", "Logan", "Avery", "Nolan", "Riley", "Carter", "Hudson", "Owen",
    "Theo", "Wyatt", "Miles", "Cole", "Jules", "Drew", "Quinn", "Reese",
    "Blake", "Hayden", "Rowan", "Casey", "Devon", "Parker", "Jamie", "Morgan",
]

FALLBACK_FAN_LAST_NAMES = [
    "Puckett", "Crossbar", "Benches", "Stickside", "Bluepaint", "Icer", "Dumpin",
    "Chase", "Sauce", "Barnburner", "Fivehole", "Overtime", "Rinkwell", "Glassman",
    "Boardley", "Slotter", "Netfront", "Clapper",
]

FAN_HANDLE_WORDS = [
    "puckwatch", "creaseburner", "neutralzone", "boardbattle", "capfriendlyish",
    "benchnoise", "slotshot", "rinkrat", "hockeypanic", "dumpandchange",
    "softdump", "powerplaymerchant", "wildtake", "goalienation", "statline",
    "forecheckfeed", "deadlinebrain", "overtimeclub", "pressboxwatch", "zoneentry",
]

FAN_PERSONAS = [
    "diehard", "homer", "skeptic", "stat nerd", "chaos fan", "old-school fan",
    "prospect watcher", "talk-radio caller", "rival fan", "playoff worrier",
    "boxscore scout", "front-office critic",
]

FAN_MARKETS = [
    "League Feed", "RinkSide", "North Stand", "Lower Bowl", "Pressbox Replies",
    "After Hours Hockey", "Puck Forum", "Fan Line", "Neutral Zone", "Late Night Thread",
]

AWARD_CATALOG: dict[str, dict[str, str]] = {
    "presidents": {"label": "Presidents' Trophy", "short": "PREZ"},
    "stanley": {"label": "Stanley Cup", "short": "CUP"},
    "art_ross": {"label": "Art Ross Trophy", "short": "ROSS"},
    "rocket": {"label": "Maurice Richard Trophy", "short": "GOAL"},
    "norris": {"label": "James Norris Memorial Trophy", "short": "NORR"},
    "hart": {"label": "Hart Memorial Trophy", "short": "HART"},
    "selke": {"label": "Frank J. Selke Trophy", "short": "SELK"},
    "calder": {"label": "Calder Memorial Trophy", "short": "CALD"},
    "vezina": {"label": "Vezina Trophy", "short": "VEZI"},
}

AWARD_ALIASES: list[tuple[str, list[str]]] = [
    ("presidents", ["presidents", "president", "presidents trophy", "presidents' trophy"]),
    ("stanley", ["stanley", "stanley cup", "cup"]),
    ("art_ross", ["art ross", "art ross trophy"]),
    ("rocket", ["rocket", "rocket richard", "maurice richard", "richard trophy"]),
    ("norris", ["norris", "norris trophy", "james norris"]),
    ("hart", ["hart", "hart memorial", "mvp"]),
    ("selke", ["selke", "frank j selke"]),
    ("calder", ["calder", "calder memorial", "rookie of the year"]),
    ("vezina", ["vezina", "vezina trophy"]),
]

AWARD_FAN_REACTION_TEMPLATES: dict[str, list[str]] = {
    "generic": [
        "{winner} winning {award} feels right. The {top_stat} number makes it hard to argue.",
        "I need the full voting breakdown, but {winner} taking {award} is not shocking at all.",
        "{award} discourse is about to be unbearable and honestly I am here for it.",
        "{winner} just added a real legacy line tonight. {legacy}",
        "People can debate the winner, but {winner} had the season everyone noticed.",
        "The finalists were strong, but {winner} always felt like the name they were building toward.",
        "That {award} race was closer than people want to admit.",
        "{winner} winning is going to age either perfectly or terribly. No middle ground.",
    ],
    "presidents": [
        "{winner} winning the Presidents' Trophy after that regular season is fair. {top_stat} says enough.",
        "Best regular-season team gets the hardware. People can hate it, but {winner} earned this.",
        "{winner} set the pace all year. Now the real pressure starts.",
        "The Presidents' Trophy is nice, but {winner} knows nobody relaxes until the playoffs are done.",
        "{winner} fans should enjoy this for about five minutes before everyone starts yelling about the curse.",
    ],
    "stanley": [
        "{winner} surviving the playoff grind and lifting the Cup is the whole story. No notes.",
        "{winner} fans are never going to let anyone forget this run.",
        "The Cup is home with {winner}. That sentence still feels insane.",
        "{winner} did not just win. They outlasted everyone.",
        "Every risky move looks genius when {winner} ends the year with the Cup.",
        "This is why you go all in. {winner} finished the job.",
    ],
    "art_ross": [
        "{winner} winning the Art Ross with {top_stat} is absurd production.",
        "Scoring race ends with {winner} on top. That tracks.",
        "{winner} led the league in points and still somehow people will call it quiet.",
        "The Art Ross going to {winner} feels like the least surprising part of Awards Night.",
        "You do not luck into the Art Ross. {winner} was a problem every night.",
    ],
    "rocket": [
        "{winner} winning the Rocket is perfect. Goal scorers are supposed to feel inevitable.",
        "{top_stat} for {winner}. That is not a hot streak, that is a season-long warning sign.",
        "Every goalie in the league is relieved {winner}'s Rocket season is finally over.",
        "{winner} owned the goal column. No overthinking needed.",
        "Goal scoring is still the loudest stat in hockey, and {winner} had the loudest season.",
    ],
    "norris": [
        "{winner} winning the Norris is going to start arguments, but the season was real.",
        "Blue line minutes, production, pressure. {winner} checked every box.",
        "{winner} taking the Norris feels like a vote for control more than just points.",
        "The Norris debate is always messy, but {winner} had the resume.",
        "If you watched {winner} every night, this Norris vote makes sense.",
    ],
    "hart": [
        "{winner} winning the Hart means the league saw what everyone else saw.",
        "Most valuable player debates are always toxic, but {winner} had the season.",
        "{winner} getting the Hart is going to make one fanbase furious and another fanbase impossible to talk to.",
        "The Hart going to {winner} feels like the headline of the entire season.",
        "Take {winner} off that team and everything looks different. That is value.",
        "This Hart race was nasty, but {winner} was always the main character.",
    ],
    "selke": [
        "{winner} winning the Selke is for the people who watch the little details.",
        "The Selke is never the loudest award, but {winner} earned it shift by shift.",
        "{winner} tilted the ice without needing every highlight to prove it.",
        "Two-way monster season from {winner}. The Selke makes sense.",
        "The box score does not always explain why {winner} won this, and that is kind of the point.",
    ],
    "calder": [
        "{winner} winning the Calder is how a fanbase starts dreaming way too early.",
        "Rookie of the year for {winner}. The future just got louder.",
        "{winner} did not look like a rookie for most of this season.",
        "The Calder going to {winner} feels like the start of something bigger.",
        "Calder winner today, unrealistic fan expectations tomorrow. That is the sport.",
    ],
    "vezina": [
        "{winner} winning the Vezina makes sense. Goaltending carried nights it had no business carrying.",
        "{top_stat} from {winner} is exactly why this Vezina vote happened.",
        "Goalie awards are chaos, but {winner} gave voters a pretty clean answer.",
        "{winner} stole enough games to make this feel obvious.",
        "The Vezina going to {winner} is a reminder that goalies still break seasons.",
    ],
}

REACTION_TONES: dict[str, list[str]] = {
    "stanley": ["celebration", "shock", "legacy", "hype"],
    "presidents": ["skeptic", "respect", "pressure", "regular-season"],
    "hart": ["debate", "legacy", "hype", "argument"],
    "vezina": ["goalie-chaos", "respect", "debate", "stolen-games"],
    "norris": ["debate", "film-room", "argument", "respect"],
    "calder": ["future", "hype", "projection", "hope"],
    "selke": ["nerd", "respect", "details", "coach-brain"],
    "rocket": ["goal-scorer", "hype", "fear", "pure-offense"],
    "art_ross": ["statline", "production", "hype", "points-race"],
}

TIME_LABELS = ["now", "12s", "28s", "45s", "1m", "2m", "4m"]


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower().replace("'", "").replace("'", "")).strip()


def _hash_string(value: Any) -> int:
    s = str(value or "seed")
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _seeded_float(seed_value: Any) -> float:
    t = (_hash_string(seed_value) + 0x6D2B79F5) & 0xFFFFFFFF
    t = ((t ^ (t >> 15)) * ((t | 1) & 0xFFFFFFFF)) & 0xFFFFFFFF
    t = (t ^ ((t + ((t ^ (t >> 7)) * ((t | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


def _seeded_int(seed_value: Any, low: int, high: int) -> int:
    if high <= low:
        return low
    return int(_seeded_float(seed_value) * (high - low + 1)) + low


def _seeded_pick(items: list[Any], seed_value: Any, fallback: str = "") -> Any:
    if not items:
        return fallback
    return items[_seeded_int(seed_value, 0, len(items) - 1)]


def _trim_tweet(text: str, max_length: int = 190) -> str:
    clean = re.sub(r"\s+([.,!?;:])", r"\1", re.sub(r"\s+", " ", str(text or "")).strip())
    if len(clean) <= max_length:
        return clean
    return f"{clean[: max_length - 1].strip()}…"


def _replace_template(template: str, values: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        val = values.get(key, "")
        return val if val else "—"

    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, template)


def resolve_award_key(raw_name: Any) -> str:
    key = _normalize_key(raw_name)
    for award_id, aliases in AWARD_ALIASES:
        for alias in aliases:
            if key == alias or alias in key:
                return award_id
    return key.replace(" ", "_") or "award"


def _award_label(award_key: str, raw_name: str = "") -> str:
    return AWARD_CATALOG.get(award_key, {}).get("label") or str(raw_name or "Award").strip()


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _round_stat(value: Any) -> str:
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "—"


def _load_awards_from_session(session: Any) -> list[dict[str, Any]]:
    payload = getattr(session, "awards_payload", None) or {}
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if not items:
        awards_map = payload.get("awards") or {}
        if isinstance(awards_map, dict):
            items = list(awards_map.values())
        else:
            items = []
    rows = [row for row in items if isinstance(row, dict) and (row.get("name") or row.get("winner_name"))]
    return rows


def _normalize_award_row(row: dict[str, Any]) -> dict[str, Any]:
    award_key = resolve_award_key(row.get("name"))
    winner_name = str(_first_defined(row.get("winner_name"), row.get("winnerName"), row.get("winner")) or "—").strip()
    winner_team = str(_first_defined(row.get("winner_team_name"), row.get("winnerTeamName")) or "").strip()
    winner_stats = row.get("winner_stats") or row.get("winnerStats") or {}
    if not isinstance(winner_stats, dict):
        winner_stats = {}

    rationale = str(_first_defined(row.get("rationale"), row.get("reason")) or "").strip()
    finalists = row.get("finalists") or row.get("candidates") or []
    if not isinstance(finalists, list):
        finalists = []

    top_stat = _top_stat_for_award(award_key, winner_stats)
    runner_up = _runner_up_label(finalists, winner_name)
    legacy = _legacy_line(row, winner_stats)
    stage_line = _award_stage_line(award_key)

    return {
        "award_key": award_key,
        "award_label": _award_label(award_key, str(row.get("name") or "")),
        "winner_label": winner_name,
        "winner_team_name": winner_team,
        "rationale": rationale or _default_rationale(award_key, winner_name),
        "legacy_line": legacy,
        "stage_line": stage_line,
        "top_stat": top_stat,
        "runner_up": runner_up,
        "finalists": finalists,
        "winner_stats": winner_stats,
    }


def _top_stat_for_award(award_key: str, stats: dict[str, Any]) -> str:
    points = _first_defined(stats.get("points"), stats.get("pts"))
    goals = _first_defined(stats.get("goals"), stats.get("g"))
    wins = _first_defined(stats.get("wins"), stats.get("w"))
    save_pct = _first_defined(stats.get("save_pct"), stats.get("savePct"), stats.get("sv_pct"))

    if award_key == "vezina" and wins is not None:
        if save_pct is not None:
            try:
                n = float(save_pct)
                sv = f"{n * 100:.1f}" if n <= 1 else f"{n:.1f}"
                return f"{_round_stat(wins)} Wins · {sv}% SV"
            except (TypeError, ValueError):
                pass
        return f"{_round_stat(wins)} Wins"
    if award_key == "rocket" and goals is not None:
        return f"{_round_stat(goals)} Goals"
    if points is not None:
        return f"{_round_stat(points)} Points"
    if goals is not None:
        return f"{_round_stat(goals)} Goals"
    if wins is not None:
        return f"{_round_stat(wins)} Wins"
    return "the numbers"


def _runner_up_label(finalists: list[Any], winner_name: str) -> str:
    for item in finalists:
        if isinstance(item, str):
            if item.strip() and item.strip() != winner_name:
                return item.strip()
            continue
        if not isinstance(item, dict):
            continue
        name = str(
            _first_defined(item.get("name"), item.get("winner_name"), item.get("full_name"), item.get("team_name"))
            or ""
        ).strip()
        if name and name != winner_name:
            votes = item.get("votes")
            if votes is not None:
                return f"{name} with {votes} vote pts"
            return name
    return "the rest of the field"


def _legacy_line(row: dict[str, Any], stats: dict[str, Any]) -> str:
    age = _first_defined(row.get("age"), stats.get("age"), stats.get("player_age"))
    if age:
        return f"Age {age}. Legacy still being written."
    if resolve_award_key(row.get("name")) == "stanley":
        return "A championship season joins franchise history."
    return "A new chapter enters league history."


def _award_stage_line(award_key: str) -> str:
    lines = {
        "presidents": "Best regular-season record.",
        "stanley": "The final team standing.",
        "art_ross": "League leader in points.",
        "rocket": "Most goals in the league.",
        "norris": "Premier defenseman.",
        "hart": "Most valuable all-around season.",
        "selke": "Elite defensive forward impact.",
        "calder": "Top first-year skater.",
        "vezina": "Best goaltending season.",
    }
    return lines.get(award_key, "Award winner announced.")


def _default_rationale(award_key: str, winner_name: str) -> str:
    lines = {
        "presidents": f"{winner_name} set the regular-season standard.",
        "stanley": f"{winner_name} survived the playoff grind and lifted the Cup.",
        "art_ross": f"{winner_name} finished as the league scoring leader.",
        "rocket": f"{winner_name} owned the goal column all season.",
        "norris": f"{winner_name} controlled play from the blue line.",
        "hart": f"{winner_name} delivered the league's defining individual season.",
        "selke": f"{winner_name} tilted the ice in every zone.",
        "calder": f"{winner_name} announced himself as a future cornerstone.",
        "vezina": f"{winner_name} gave his team elite goaltending all year.",
    }
    return lines.get(award_key, f"{winner_name} takes home season hardware.")


def _compact_handle_part(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", str(value or "").lower())[:20]


def _build_fan_handle(first: str, last: str, seed: str) -> str:
    word = _seeded_pick(FAN_HANDLE_WORDS, f"{first}{last}:{seed}:word", "puckwatch")
    number = _seeded_int(f"{first}{last}:{seed}:number", 11, 989)
    compact_first = _compact_handle_part(first)[:10]
    compact_last = _compact_handle_part(last)[:10]
    styles = [
        f"@{word}{number}",
        f"@{compact_first}{number}",
        f"@{word}{compact_last[:6]}{number % 100}",
        f"@{compact_first}_{compact_last[:8]}{number % 1000}",
    ]
    return _seeded_pick(styles, f"{first}{last}:{seed}:style", f"@{word}{number}")


def _normalize_fan_profile(raw: dict[str, Any], index: int, seed: str) -> dict[str, str]:
    first = str(
        _first_defined(
            raw.get("first"),
            (raw.get("name") or {}).get("first") if isinstance(raw.get("name"), dict) else None,
        )
        or _seeded_pick(FALLBACK_FAN_FIRST_NAMES, f"{seed}:{index}:first", "Rink")
    ).strip()
    last = str(
        _first_defined(
            raw.get("last"),
            (raw.get("name") or {}).get("last") if isinstance(raw.get("name"), dict) else None,
        )
        or _seeded_pick(FALLBACK_FAN_LAST_NAMES, f"{seed}:{index}:last", "Watcher")
    ).strip()
    display_name = f"{first} {last}".strip()
    login = raw.get("login") if isinstance(raw.get("login"), dict) else {}
    handle = str(_first_defined(raw.get("handle"), login.get("username")) or "").strip()
    if not handle.startswith("@"):
        handle = _build_fan_handle(first, last, f"{seed}:{index}")
    picture = raw.get("picture") if isinstance(raw.get("picture"), dict) else {}
    avatar_src = str(
        _first_defined(raw.get("avatar_src"), raw.get("avatarSrc"), picture.get("thumbnail"), picture.get("large"))
        or ""
    ).strip()
    nat = ""
    nat_raw = raw.get("nat")
    if isinstance(nat_raw, dict):
        nat = str(nat_raw.get("code") or nat_raw.get("nationality") or "").strip().upper()
    elif nat_raw:
        nat = str(nat_raw).strip().upper()

    fan_id = str(
        _first_defined(raw.get("id"), login.get("uuid")) or f"fan-{_hash_string(f'{seed}:{index}:{display_name}')}"
    )

    return {
        "id": fan_id if fan_id.startswith("fan_") else f"fan_{fan_id}",
        "display_name": display_name,
        "handle": handle,
        "avatar_src": avatar_src,
        "persona": str(raw.get("persona") or _seeded_pick(FAN_PERSONAS, f"{seed}:{index}:persona", "fan")),
        "market": str(raw.get("market") or _seeded_pick(FAN_MARKETS, f"{seed}:{index}:market", "League Feed")),
        "nat": nat or _seeded_pick(["CA", "US", "SE", "FI", "CH"], f"{seed}:{index}:nat", "CA"),
    }


def _build_fallback_fans(count: int, seed: str) -> list[dict[str, str]]:
    total = max(1, min(int(count), 80))
    fans: list[dict[str, str]] = []
    for index in range(total):
        first = _seeded_pick(FALLBACK_FAN_FIRST_NAMES, f"{seed}:{index}:first", "Rink")
        last = _seeded_pick(FALLBACK_FAN_LAST_NAMES, f"{seed}:{index}:last", "Watcher")
        fans.append(
            _normalize_fan_profile(
                {
                    "id": f"fallback-fan-{index}",
                    "first": first,
                    "last": last,
                    "handle": _build_fan_handle(first, last, f"{seed}:{index}"),
                    "avatar_src": "",
                },
                index,
                seed,
            )
        )
    return fans


def _fetch_external_fans(fan_count: int, seed: str, timeout: float = 4.0) -> list[dict[str, str]]:
    params = urlencode(
        {
            "results": max(1, min(int(fan_count), 80)),
            "seed": seed,
            "inc": "name,login,picture,nat",
            "noinfo": "true",
        }
    )
    url = f"{RANDOM_FAN_API_URL}?{params}"
    request = Request(url, headers={"User-Agent": "NHL-Franchise-Mode/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    fans = [_normalize_fan_profile(row, index, seed) for index, row in enumerate(results) if isinstance(row, dict)]
    return fans


def _fetch_fan_pool(fan_count: int, seed: str, use_external: bool) -> list[dict[str, str]]:
    if use_external:
        try:
            fans = _fetch_external_fans(fan_count, seed)
            if fans:
                return fans
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
            pass
    return _build_fallback_fans(fan_count, seed)


def _context_values(award: dict[str, Any]) -> dict[str, str]:
    return {
        "award": award.get("award_label") or "the award",
        "winner": award.get("winner_label") or "the winner",
        "winner_team": award.get("winner_team_name") or "their team",
        "top_stat": award.get("top_stat") or "the numbers",
        "runner_up": award.get("runner_up") or "the rest of the field",
        "legacy": award.get("legacy_line") or "The legacy keeps building.",
        "rationale": award.get("rationale") or "The case was strong.",
        "stage_line": award.get("stage_line") or "Award winner announced.",
    }


def _reaction_tone(award_key: str, seed: str) -> str:
    tones = REACTION_TONES.get(award_key, ["reaction", "debate", "hype"])
    return str(_seeded_pick(tones, f"{award_key}:{seed}:tone", "reaction"))


def _build_tweet_metrics(seed: str) -> dict[str, int]:
    likes = _seeded_int(f"{seed}:likes", 8, 980)
    reposts = _seeded_int(f"{seed}:reposts", 0, max(4, likes // 5))
    replies = _seeded_int(f"{seed}:replies", 0, max(3, likes // 8))
    quotes = _seeded_int(f"{seed}:quotes", 0, max(2, likes // 12))
    return {"replies": replies, "reposts": reposts, "quotes": quotes, "likes": likes}


def _build_reaction_text(award: dict[str, Any], fan: dict[str, str], index: int, seed: str) -> str:
    award_key = award.get("award_key") or "generic"
    templates = list(AWARD_FAN_REACTION_TEMPLATES.get(award_key, [])) + list(
        AWARD_FAN_REACTION_TEMPLATES.get("generic", [])
    )
    template = _seeded_pick(
        templates,
        f"{seed}:{award_key}:{award.get('winner_label')}:{fan.get('handle')}:{index}:template",
        "{winner} wins {award}. The discourse starts now.",
    )
    return _trim_tweet(_replace_template(template, _context_values(award)), 190)


def _tweets_per_award(count: int, award_total: int) -> int:
    if award_total <= 0:
        return 1
    if count >= award_total * 3:
        return 3
    if count >= award_total * 2:
        return 2
    return 1


def _build_awards_fan_reactions(
    session: Any,
    *,
    count: int = 18,
    fan_count: int = 24,
    seed: Optional[str] = None,
    award_key: Optional[str] = None,
    use_external_fans: bool = True,
) -> dict[str, Any]:
    session_id = str(getattr(session, "session_id", "") or "")
    season_year = getattr(session, "season_calendar_year", None) or getattr(session, "season_year", None) or ""
    resolved_seed = str(seed or f"{session_id}:{season_year}:awards-night").strip() or "awards-night"
    max_tweets = max(1, min(int(count), 40))
    max_fans = max(1, min(int(fan_count), 80))

    raw_awards = _load_awards_from_session(session)
    awards = [_normalize_award_row(row) for row in raw_awards]
    if award_key:
        awards = [a for a in awards if a.get("award_key") == award_key]

    fan_pool = _fetch_fan_pool(max_fans, resolved_seed, use_external_fans)
    tweets: list[dict[str, Any]] = []

    if not awards:
        fan = fan_pool[0] if fan_pool else _normalize_fan_profile({}, 0, resolved_seed)
        metric_seed = f"{resolved_seed}:empty"
        tweets.append(
            {
                "id": f"tweet_{_hash_string(metric_seed)}",
                "type": "award_reaction",
                "award_key": "awards",
                "award_label": "Awards Night",
                "winner_label": "",
                "winner_team_name": "",
                "text": "Awards Night is open, but no winners have hit the feed yet.",
                "tone": "empty",
                "created_at_label": "now",
                "fan": fan,
                "metrics": _build_tweet_metrics(metric_seed),
                "context": {
                    "top_stat": "",
                    "runner_up": "",
                    "legacy": "",
                },
            }
        )
    else:
        per_award = _tweets_per_award(max_tweets, len(awards))
        for award_index, award in enumerate(awards):
            for i in range(per_award):
                fan_index = (award_index * per_award + i + 1) % len(fan_pool)
                fan = fan_pool[fan_index]
                metric_seed = f"{resolved_seed}:{award['award_key']}:{award['winner_label']}:{fan['handle']}:{i}"
                tweets.append(
                    {
                        "id": f"tweet_{_hash_string(metric_seed)}",
                        "type": "award_reaction",
                        "award_key": award["award_key"],
                        "award_label": award["award_label"],
                        "winner_label": award["winner_label"],
                        "winner_team_name": award.get("winner_team_name") or "",
                        "text": _build_reaction_text(award, fan, i, resolved_seed),
                        "tone": _reaction_tone(award["award_key"], metric_seed),
                        "created_at_label": _seeded_pick(TIME_LABELS, f"{metric_seed}:time", "now"),
                        "fan": fan,
                        "metrics": _build_tweet_metrics(metric_seed),
                        "context": {
                            "top_stat": award.get("top_stat") or "",
                            "runner_up": award.get("runner_up") or "",
                            "legacy": award.get("legacy_line") or "",
                        },
                    }
                )

    tweets = tweets[:max_tweets]
    total_likes = sum(int(t.get("metrics", {}).get("likes") or 0) for t in tweets)
    debate_keys = {"hart", "norris", "vezina", "calder", "selke"}
    debate_awards = [a for a in awards if a.get("award_key") in debate_keys]
    hottest = (
        debate_awards[0]
        if debate_awards
        else next((a for a in awards if a.get("award_key") == "stanley"), None)
        or (awards[0] if awards else None)
    )

    pulse_headline = (
        f"{hottest['winner_label']} has the feed moving"
        if hottest and hottest.get("winner_label")
        else "Awards Night has the feed moving"
    )
    pulse_subline = (
        f"{hottest['award_label']} is driving the loudest reactions."
        if hottest and hottest.get("award_label")
        else "Fans are reacting across the league."
    )

    return {
        "kind": "awards_fan_reactions",
        "session_id": session_id,
        "seed": resolved_seed,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "tweets": tweets,
        "pulse": {
            "headline": pulse_headline,
            "subline": pulse_subline,
            "total_likes": total_likes,
            "hottest_award_key": (hottest or {}).get("award_key") or "",
        },
    }


def register_fan_reactions_routes(app: Any, session_or_404: Callable[[Optional[str]], Any]) -> None:
    """Attach fan reaction routes to the FastAPI app."""

    @app.get("/api/franchise/{session_id}/fan-reactions/awards")
    def get_awards_fan_reactions(
        session_id: str,
        count: int = 18,
        fan_count: int = 24,
        seed: Optional[str] = None,
        award_key: Optional[str] = None,
        use_external_fans: bool = False,
        event_type: Optional[str] = None,
    ) -> dict[str, Any]:
        session = session_or_404(session_id)
        if str(getattr(session, "session_id", "")) != str(session_id):
            raise HTTPException(status_code=404, detail="Unknown or expired franchise session")
        # Draft reactions are generated client-side from pick payloads via the shared
        # Award Show universe (awardHelpers.buildDraftFanTweets). Backend awards path unchanged.
        if str(event_type or "").lower() in ("entry_draft", "draft", "draft.selection"):
            return {
                "kind": "draft_fan_reactions",
                "session_id": session_id,
                "seed": seed or f"{getattr(session, 'season_calendar_year', '')}-entry-draft",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tweets": [],
                "pulse": {"note": "Use client buildDraftFanTweets with completed_picks"},
                "event_type": "entry_draft",
            }
        return _build_awards_fan_reactions(
            session,
            count=count,
            fan_count=fan_count,
            seed=seed,
            award_key=award_key,
            use_external_fans=use_external_fans,
        )
