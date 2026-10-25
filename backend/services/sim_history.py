"""PuckGPT: one reader and one question engine for a franchise save.

Games, lineups, and loud social posts are copied into ``session.puck_archive``
when they happen. The box-score book keeps every season, and this archive
reads that book so a question asked in 2045 can still name February 15, 2029.

This is not a model call. It reads the save and answers in plain language.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_MAX_LOG = 4000
_MAX_GAMES = 40000
_MAX_SOCIAL = 8000

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_STOP = {
    "the", "who", "what", "show", "this", "that", "with", "from", "have", "has",
    "played", "play", "game", "games", "player", "players", "about", "which",
    "were", "was", "did", "does", "best", "worst", "social", "media", "said",
    "across", "league", "fit", "team", "for", "and", "any",
}


def append_sim_history(session: Any, kind: str, summary: str, **fields: Any) -> None:
    if session is None:
        return
    log = getattr(session, "sim_history", None)
    if not isinstance(log, list):
        log = []
        try:
            session.sim_history = log
        except Exception:
            return
    event: Dict[str, Any] = {
        "kind": str(kind or "note"),
        "summary": str(summary or ""),
        "season_year": int(getattr(session, "season_calendar_year", 0) or 0),
    }
    for key, value in fields.items():
        if value is None:
            continue
        event[str(key)] = value
    log.append(event)
    extra = len(log) - _MAX_LOG
    if extra > 0:
        del log[:extra]


def _archive(session: Any) -> Dict[str, Any]:
    arch = getattr(session, "puck_archive", None)
    if not isinstance(arch, dict):
        arch = {"games": [], "seen": [], "social": []}
        try:
            session.puck_archive = arch
        except Exception:
            return {"games": [], "seen": [], "social": []}
    arch.setdefault("games", [])
    arch.setdefault("seen", [])
    arch.setdefault("social", [])
    return arch


def _remember_game(session: Any, game: Dict[str, Any]) -> None:
    gid = str(game.get("id") or "")
    if not gid or not game.get("date"):
        return
    arch = _archive(session)
    seen = arch["seen"]
    if gid in seen:
        return
    seen.append(gid)
    arch["games"].append(game)
    overflow = len(arch["games"]) - _MAX_GAMES
    if overflow > 0:
        drop = {str(row.get("id") or "") for row in arch["games"][:overflow]}
        del arch["games"][:overflow]
        arch["seen"] = [item for item in seen if item not in drop]


def absorb_box_book(session: Any) -> None:
    """Copy the live box-score book into the long archive before a season is dropped."""
    book = getattr(session, "box_score_book", None)
    if not isinstance(book, dict):
        return
    seasons = book.get("seasons") or {}
    if not isinstance(seasons, dict):
        return
    for bucket in seasons.values():
        if not isinstance(bucket, dict):
            continue
        people = bucket.get("people") or {}
        games = bucket.get("games") or {}
        if not isinstance(games, dict):
            continue
        for gid, compact in games.items():
            if not isinstance(compact, dict):
                continue
            names = _names_from_compact(compact, people if isinstance(people, dict) else {})
            _remember_game(session, {
                "id": str(compact.get("id") or gid),
                "date": str(compact.get("iso") or "")[:10],
                "home": compact.get("hn") or compact.get("home") or "Home",
                "away": compact.get("an") or compact.get("away") or "Away",
                "hg": compact.get("hg"),
                "ag": compact.get("ag"),
                "players": names,
                "ot": bool(compact.get("ot")),
                "so": bool(compact.get("so")),
            })


def _names_from_compact(compact: Dict[str, Any], people: Dict[str, Any]) -> List[str]:
    lines = compact.get("lines") or {}
    if not isinstance(lines, dict):
        return []
    names: List[str] = []
    for pid in lines:
        person = people.get(pid) or people.get(str(pid))
        if isinstance(person, (list, tuple)) and person:
            label = str(person[0] or "").strip()
        elif isinstance(person, str):
            label = person.strip()
        else:
            label = ""
        if label and label.lower() != "player":
            names.append(label)
        if len(names) >= 40:
            break
    return names


def _names_from_lines(session: Any, lines: Any) -> List[str]:
    if not isinstance(lines, dict):
        return []
    ledger = getattr(session, "player_season_stats", None) or {}
    names: List[str] = []
    for pid, row in lines.items():
        label = ""
        if isinstance(row, dict):
            label = str(row.get("name") or "").strip()
        if not label and isinstance(ledger, dict):
            stat = ledger.get(pid) or ledger.get(str(pid)) or {}
            if isinstance(stat, dict):
                label = str(stat.get("name") or "").strip()
        if label:
            names.append(label)
        if len(names) >= 40:
            break
    return names


def record_final_score(session: Any, box: Dict[str, Any], lines: Any = None) -> None:
    if not isinstance(box, dict) or session is None:
        return
    try:
        absorb_box_book(session)
    except Exception:
        pass
    home = box.get("home_name") or box.get("home_abbr") or box.get("home_id") or "Home"
    away = box.get("away_name") or box.get("away_abbr") or box.get("away_id") or "Away"
    hg = box.get("home_goals", box.get("home_score"))
    ag = box.get("away_goals", box.get("away_score"))
    extra = ""
    if box.get("shootout"):
        extra = " (SO)"
    elif box.get("overtime") or box.get("ot"):
        extra = " (OT)"
    date = str(box.get("calendar_iso") or box.get("iso") or "")[:10]
    gid = str(box.get("game_id") or box.get("id") or f"{date}:{away}@{home}:{hg}-{ag}")
    players = _names_from_lines(session, lines)
    _remember_game(session, {
        "id": gid,
        "date": date,
        "home": home,
        "away": away,
        "hg": hg,
        "ag": ag,
        "players": players,
        "ot": bool(box.get("overtime") or box.get("ot")),
        "so": bool(box.get("shootout")),
    })
    played = f" Played: {', '.join(players[:18])}." if players else ""
    append_sim_history(
        session,
        "game",
        f"{date} {away} {ag} @ {home} {hg}{extra}.{played}".strip(),
        home_id=box.get("home_id"),
        away_id=box.get("away_id"),
        home_goals=hg,
        away_goals=ag,
        overtime=bool(box.get("overtime") or box.get("ot")),
        shootout=bool(box.get("shootout")),
        playoff=bool(box.get("is_playoff")),
        date=date,
        players=players[:24],
    )


def archive_social_line(session: Any, post: Dict[str, Any]) -> None:
    """Keep a loud or player-written post after the live feed prunes it."""
    if session is None or not isinstance(post, dict):
        return
    try:
        sentiment = float(post.get("sentiment") or 0)
    except (TypeError, ValueError):
        sentiment = 0.0
    kind = str(post.get("kind") or "")
    player = str(post.get("player_name") or post.get("author_name") or "").strip()
    if not player:
        return
    if abs(sentiment) < 0.35 and not kind.startswith("player"):
        return
    arch = _archive(session)
    arch["social"].append({
        "date": str(post.get("calendar_iso") or "")[:10],
        "name": player,
        "text": str(post.get("text") or "")[:220],
        "sentiment": round(sentiment, 2),
        "kind": kind,
    })
    extra = len(arch["social"]) - _MAX_SOCIAL
    if extra > 0:
        del arch["social"][:extra]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_games(session: Any) -> List[Dict[str, Any]]:
    try:
        absorb_box_book(session)
    except Exception:
        pass
    arch = _archive(session)
    games = [row for row in arch.get("games") or [] if isinstance(row, dict)]
    log = getattr(session, "sim_history", None) or []
    seen = {str(row.get("id") or "") for row in games}
    for row in log:
        if not isinstance(row, dict) or row.get("kind") != "game":
            continue
        gid = str(row.get("date") or "") + str(row.get("summary") or "")
        if gid in seen:
            continue
        games.append({
            "id": gid,
            "date": str(row.get("date") or "")[:10],
            "home": "",
            "away": "",
            "hg": row.get("home_goals"),
            "ag": row.get("away_goals"),
            "players": list(row.get("players") or []),
            "summary": row.get("summary") or "",
            "ot": bool(row.get("overtime")),
            "so": bool(row.get("shootout")),
        })
    return games


def _read_social(session: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for post in list(getattr(session, "social_posts", None) or []):
        if not isinstance(post, dict):
            continue
        name = str(post.get("player_name") or post.get("author_name") or "").strip()
        if not name:
            continue
        try:
            sentiment = float(post.get("sentiment") or 0)
        except (TypeError, ValueError):
            sentiment = 0.0
        rows.append({
            "date": str(post.get("calendar_iso") or "")[:10],
            "name": name,
            "text": str(post.get("text") or "")[:220],
            "sentiment": sentiment,
            "kind": str(post.get("kind") or ""),
        })
    arch = getattr(session, "puck_archive", None) or {}
    for row in list(arch.get("social") or []) if isinstance(arch, dict) else []:
        if isinstance(row, dict) and row.get("name"):
            rows.append(row)
    entities = getattr(session, "universe_players", None) or {}
    if isinstance(entities, dict):
        for ent in entities.values():
            if not isinstance(ent, dict):
                continue
            social = ent.get("social") if isinstance(ent.get("social"), dict) else {}
            if social.get("fan_sentiment") is None:
                continue
            try:
                mood = (float(social.get("fan_sentiment")) - 55.0) / 45.0
            except (TypeError, ValueError):
                continue
            name = str(ent.get("player_name") or "").strip()
            if name:
                rows.append({
                    "date": "",
                    "name": name,
                    "text": "Public mood on his account.",
                    "sentiment": max(-1.0, min(1.0, mood)),
                    "kind": "mood",
                })
    return rows


def _player_ovr(player: Any) -> float:
    raw = getattr(player, "overall", None)
    if raw is None:
        raw = getattr(player, "ovr", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = 0
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    if 0 < value <= 1.5:
        value *= 99.0
    return value


def _player_pot(player: Any) -> float:
    ratings = getattr(player, "ratings", None)
    raw = None
    if isinstance(ratings, dict):
        raw = ratings.get("dev_potential")
    if raw in (None, 0, 0.0):
        raw = getattr(player, "potential", None)
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    if 0 < value <= 1.5:
        value *= 99.0
    return value


def _player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return str(
        getattr(ident, "full_name", None)
        or getattr(ident, "name", None)
        or getattr(player, "name", None)
        or "Player"
    )


def _player_pos(player: Any) -> str:
    ident = getattr(player, "identity", None)
    raw = getattr(ident, "position", None) if ident is not None else None
    if raw is None:
        raw = getattr(player, "position", "")
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw or "").upper()


def _player_age(player: Any) -> int:
    ident = getattr(player, "identity", None)
    raw = getattr(ident, "age", None) if ident is not None else None
    if raw is None:
        raw = getattr(player, "age", 0)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _iter_roster(session: Any) -> Iterable[Tuple[str, Any, Any]]:
    teams = getattr(session, "team_by_id", None) or {}
    if not isinstance(teams, dict):
        return
    for tid, team in teams.items():
        for player in list(getattr(team, "roster", None) or []):
            yield str(tid), team, player


def _team_label(team: Any, tid: str) -> str:
    return str(getattr(team, "abbreviation", None) or getattr(team, "abbr", None) or getattr(team, "name", None) or tid)


def _asked_position(query: str) -> str:
    q = f" {query.lower()} "
    if any(token in q for token in (" ld", " lhd", " left defense", " left d")):
        return "LD"
    if any(token in q for token in (" rd", " rhd", " right defense", " right d")):
        return "RD"
    if "goalie" in q or " goaltender" in q or " g " in q:
        return "G"
    if "center" in q or " centre" in q:
        return "C"
    if "left wing" in q or " lw" in q:
        return "LW"
    if "right wing" in q or " rw" in q:
        return "RW"
    if "defense" in q or "defence" in q or " d " in q or "pair" in q:
        return "D"
    return ""


def _pos_matches(pos: str, want: str) -> bool:
    pos = pos.upper()
    if want == "D":
        return pos in ("D", "LD", "RD", "LHD", "RHD") or pos.startswith("D")
    if want == "LD":
        return pos in ("LD", "LHD", "D")
    if want == "RD":
        return pos in ("RD", "RHD", "D")
    if want == "G":
        return pos in ("G", "GOALIE")
    return pos == want or pos.startswith(want)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def _parse_date(query: str) -> Optional[str]:
    text = query.lower().replace(",", " ")
    iso = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", text)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    month_first = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s+(20\d{2})",
        text,
    )
    if month_first:
        month = _MONTHS[month_first.group(1)[:3]]
        return f"{int(month_first.group(3)):04d}-{month:02d}-{int(month_first.group(2)):02d}"
    day_first = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(20\d{2})",
        text,
    )
    if day_first:
        month = _MONTHS[day_first.group(2)[:3]]
        return f"{int(day_first.group(3)):04d}-{month:02d}-{int(day_first.group(1)):02d}"
    return None


def _games_on(session: Any, date: str) -> Dict[str, Any]:
    hits = []
    for game in _read_games(session):
        if str(game.get("date") or "")[:10] != date:
            continue
        players = [str(name) for name in (game.get("players") or []) if name]
        home = game.get("home") or "Home"
        away = game.get("away") or "Away"
        mark = " SO" if game.get("so") else (" OT" if game.get("ot") else "")
        if game.get("summary") and not home:
            summary = str(game.get("summary"))
        else:
            summary = f"{away} {game.get('ag')} @ {home} {game.get('hg')}{mark}"
        if players:
            summary += ". Played: " + ", ".join(players[:24])
        else:
            summary += ". The score is on file. This game was stored without a lineup."
        hits.append({"kind": "game", "date": date, "summary": summary})
    if not hits:
        return {
            "query": date,
            "answer": f"No game on {date} is in this save. Lineups are kept from the day this desk started recording.",
            "hits": [],
        }
    return {
        "query": date,
        "answer": f"{len(hits)} game{'s' if len(hits) != 1 else ''} on {date}.",
        "hits": hits[:20],
    }


def _social_rank(session: Any, query: str) -> Dict[str, Any]:
    worst = "worst" in query or "bad" in query or "negative" in query
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in _read_social(session):
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        buckets.setdefault(name, []).append(row)
    scored = []
    for name, posts in buckets.items():
        moods = [float(p.get("sentiment") or 0) for p in posts]
        quote = max(posts, key=lambda p: abs(float(p.get("sentiment") or 0)))
        scored.append((sum(moods) / len(moods), name, quote, len(posts)))
    scored.sort(key=lambda item: item[0], reverse=not worst)
    picked = scored[:8]
    if not picked:
        return {
            "query": query,
            "answer": "No social posts are in this save yet. They show up here after the feed writes them.",
            "hits": [],
        }
    label = "coldest" if worst else "warmest"
    hits = []
    for mood, name, quote, count in picked:
        text = str(quote.get("text") or "").strip()
        when = quote.get("date") or ""
        hits.append({
            "kind": "social",
            "date": when,
            "summary": f"{name} ({mood:+.2f} across {count}): {text}",
        })
    return {
        "query": query,
        "answer": f"The {label} voices in the social archive.",
        "hits": hits,
    }


def _fit_answer(session: Any, query: str) -> Dict[str, Any]:
    want = _asked_position(query)
    if not want:
        want = "D" if "pair" in query else ""
    if not want:
        return {
            "query": query,
            "answer": "Name a spot: second-pair LD, a center, a goalie.",
            "hits": [],
        }
    user_id = str(getattr(session, "user_team_id", "") or "")
    mine: List[Tuple[float, str, str]] = []
    others: List[Tuple[float, str, str, int, float]] = []
    for tid, team, player in _iter_roster(session):
        pos = _player_pos(player)
        if not _pos_matches(pos, want):
            continue
        ovr = _player_ovr(player)
        pot = _player_pot(player)
        age = _player_age(player)
        name = _player_name(player)
        club = _team_label(team, tid)
        if tid == user_id:
            mine.append((ovr, name, pos))
        elif age <= 32:
            others.append((pot or ovr, name, club, age, ovr))
    mine.sort(reverse=True)
    others.sort(reverse=True)
    own = ", ".join(f"{name} {ovr:.0f}" for ovr, name, _pos in mine[:6]) or "nobody at that spot"
    hits = []
    for pot, name, club, age, ovr in others[:8]:
        hits.append({
            "kind": "fit",
            "summary": f"{name}, {age}, {club} — {ovr:.0f} overall, {pot:.0f} potential",
        })
    chair = "second pair" if "pair" in query else want
    return {
        "query": query,
        "answer": f"Your {chair} group: {own}. These are the best fits outside the room.",
        "hits": hits,
    }


def _keyword_search(session: Any, query: str) -> Dict[str, Any]:
    words = [
        w for w in re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
        if len(w) > 2 and w not in _STOP
    ]
    hits = []
    for game in reversed(_read_games(session)):
        blob = " ".join(str(game.get(k) or "") for k in ("date", "home", "away", "summary")).lower()
        blob += " " + " ".join(str(n).lower() for n in (game.get("players") or []))
        if words and any(w in blob for w in words):
            summary = str(game.get("summary") or "")
            if not summary:
                summary = f"{game.get('date') or ''} {game.get('away')} {game.get('ag')} @ {game.get('home')} {game.get('hg')}".strip()
            hits.append({"kind": "game", "date": game.get("date") or "", "summary": summary})
        if len(hits) >= 12:
            break
    log = [row for row in (getattr(session, "sim_history", None) or []) if isinstance(row, dict)]
    if hits:
        return {"query": query, "answer": hits[0]["summary"], "hits": hits}
    if "win streak" in query or "losing streak" in query:
        return {"query": query, "answer": _streak_answer(log, losing="losing" in query), "hits": []}
    return {
        "query": query,
        "answer": f"Nothing in the archive matches that ({len(_read_games(session))} games on file).",
        "hits": log[-6:],
    }


def _streak_answer(log: List[Dict[str, Any]], *, losing: bool) -> str:
    best = 0
    run = 0
    for row in log:
        if row.get("kind") != "game":
            continue
        try:
            hg = int(row.get("home_goals"))
            ag = int(row.get("away_goals"))
        except (TypeError, ValueError):
            continue
        home_won = hg > ag
        won = (not home_won) if losing else home_won
        run = run + 1 if won else 0
        best = max(best, run)
    label = "losing" if losing else "winning"
    if best <= 0:
        return f"No {label} streak is in the log yet."
    return f"Longest {label} run currently in the log is {best} games."


def answer_save_query(session: Any, query: str) -> Dict[str, Any]:
    """Answer one question from the archive, the box book, the feed, and the rosters."""
    q = " ".join(str(query or "").lower().split())
    if not q:
        return {"query": query, "answer": "Ask who played on a date, who was loudest on social, or who fits a spot.", "hits": []}
    date = _parse_date(q)
    if date and any(word in q for word in ("play", "played", "game", "games", "who", "score", "lineup")):
        return _games_on(session, date)
    if date and not any(word in q for word in ("social", "fit", "trade")):
        return _games_on(session, date)
    if "social" in q or "tweet" in q or "post" in q:
        return _social_rank(session, q)
    if "fit" in q or "second pair" in q or "second-pair" in q:
        return _fit_answer(session, q)
    return _keyword_search(session, q)
