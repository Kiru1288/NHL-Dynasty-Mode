"""Unified player search across every level of the hockey world.

NHL rosters + IR, AHL/ECHL farm rosters, NHL prospect pools (rights held),
development leagues (CHL, NCAA, Europe juniors), and the free-agent pools.
Also resolves any player id to a dossier-ready row so a single click from
Stats Central, the prospect ledger, or search opens the Player Dossier.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Tuple

_log = logging.getLogger(__name__)

# (player, level tag, NHL org team obj or None, club label, league code)
_Entry = Tuple[Any, str, Optional[Any], str, str]


def _fold(text: Any) -> str:
    s = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", "") or "")


def _team_label(t: Any) -> str:
    return str(getattr(t, "abbreviation", None) or getattr(t, "abbr", None) or getattr(t, "name", "") or "").upper()


def _pid(p: Any) -> str:
    raw = str(getattr(p, "id", "") or getattr(p, "player_id", "") or "").strip()
    return f"NHL_{raw}" if raw.isdigit() else raw


def iter_all_players(session: Any) -> Iterator[_Entry]:
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return
    seen: set = set()

    def _emit(p: Any, level: str, team: Any, club: str, code: str) -> Optional[_Entry]:
        if p is None or getattr(p, "retired", False):
            return None
        key = _pid(p) or str(id(p))
        if key in seen:
            return None
        seen.add(key)
        return (p, level, team, club, code)

    for t in list(getattr(league, "teams", None) or []):
        abbr = _team_label(t)
        for attr, level in (
            ("roster", "NHL"),
            ("injured_reserve", "NHL"),
            ("ahl_roster", "AHL"),
            ("echl_roster", "ECHL"),
        ):
            for p in list(getattr(t, attr, None) or []):
                e = _emit(p, level, t, abbr, level)
                if e:
                    yield e
        for p in list(getattr(t, "prospect_pool", None) or getattr(t, "prospects", None) or []):
            asg = getattr(p, "_franchise_assignment", None) or {}
            club = str(asg.get("club") or "") if isinstance(asg, dict) else ""
            code = str(asg.get("league_code") or "") if isinstance(asg, dict) else ""
            e = _emit(p, "PROSPECT", t, club or abbr, code)
            if e:
                yield e
    teams_by_id = getattr(session, "team_by_id", None) or {}
    for block in list(getattr(league, "development_leagues", None) or []):
        if not isinstance(block, dict):
            continue
        code = str(block.get("league_code") or "")
        for tm in block.get("teams") or []:
            if not isinstance(tm, dict):
                continue
            club = str(tm.get("name") or tm.get("team_name") or "")
            for p in tm.get("players") or []:
                rid = str(getattr(p, "nhl_rights_team_id", None) or getattr(p, "rights_team_id", None) or "")
                org = teams_by_id.get(rid) if rid and not getattr(p, "_bootstrap_fake_rights", False) else None
                e = _emit(p, "JUNIOR", org, club, code)
                if e:
                    yield e
    for attr, level in (("free_agents", "FA"), ("overseas_free_agents", "FA-EU")):
        for p in list(getattr(league, attr, None) or []):
            e = _emit(p, level, None, "Free agent", level)
            if e:
                yield e


def _ovr(p: Any) -> float:
    f = getattr(p, "ovr", None)
    try:
        v = float(f() if callable(f) else f)
    except Exception:
        return 0.0
    return round(v * 100.0, 1) if v <= 1.0 else round(v, 1)


def _age(p: Any) -> int:
    ident = getattr(p, "identity", None)
    try:
        return int(getattr(ident, "age", None) or getattr(p, "age", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _pos(p: Any) -> str:
    ident = getattr(p, "identity", None)
    raw = getattr(ident, "position", None) or getattr(p, "position", "")
    return str(getattr(raw, "value", raw) or "").upper()


def _league_label(code: str) -> str:
    try:
        from services.franchise_sim import _draft_league_display

        return _draft_league_display(code)
    except Exception:
        return str(code or "")


def search_players(session: Any, q: str, limit: int = 40) -> Dict[str, Any]:
    needle = _fold(q)
    if len(needle) < 2:
        return {"query": q, "results": [], "count": 0}
    terms = needle.split()
    hits: List[Tuple[int, float, Dict[str, Any]]] = []
    for p, level, team, club, code in iter_all_players(session):
        name = _fold(_name(p))
        if not name or not all(t in name for t in terms):
            continue
        last = name.split()[-1] if name.split() else name
        # Exact > last-name prefix > first/any-word prefix > substring.
        if name == needle:
            rank = 0
        elif last.startswith(needle):
            rank = 1
        elif any(w.startswith(terms[0]) for w in name.split()):
            rank = 2
        else:
            rank = 3
        ovr = _ovr(p)
        hits.append(
            (
                rank,
                -ovr,
                {
                    "player_id": _pid(p),
                    "name": _name(p),
                    "age": _age(p),
                    "position": _pos(p),
                    "ovr": ovr,
                    "level": level,
                    "league_code": code,
                    "league": level if level in ("NHL", "AHL", "ECHL", "FA", "FA-EU") else (_league_label(code) or level),
                    "club": club,
                    "org_team_id": str(getattr(team, "team_id", None) or getattr(team, "id", "") or "") if team is not None else "",
                    "org_abbr": _team_label(team) if team is not None else "",
                },
            )
        )
    hits.sort(key=lambda h: (h[0], h[1]))
    rows = [h[2] for h in hits[: max(1, min(int(limit or 40), 100))]]
    return {"query": q, "results": rows, "count": len(hits)}


def find_player_entry(session: Any, player_id: str) -> Optional[_Entry]:
    want = str(player_id or "").strip()
    if not want:
        return None
    alt = want[4:] if want.startswith("NHL_") else f"NHL_{want}" if want.isdigit() else want
    for entry in iter_all_players(session):
        pid = _pid(entry[0])
        raw = str(getattr(entry[0], "id", "") or "")
        if want in (pid, raw) or alt in (pid, raw):
            return entry
    return None


def player_dossier_row(session: Any, player_id: str) -> Dict[str, Any]:
    """Roster-browser-shaped row for any player id, so the dossier opens anywhere."""
    entry = find_player_entry(session, player_id)
    if entry is None:
        return {"found": False, "player_id": str(player_id)}
    p, level, team, club, code = entry
    from services.franchise_sim import _serialize_player_row

    kind = {"AHL": "ahl", "ECHL": "echl"}.get(level, "")
    try:
        row = _serialize_player_row(p, include_ratings=True, session=session, _team=team, roster_kind=kind)
    except Exception:
        _log.exception("dossier row serialize failed for %s", player_id)
        row = {"player_id": _pid(p), "name": _name(p), "age": _age(p), "position": _pos(p), "ovr": _ovr(p)}
    row = dict(row)
    row.setdefault("player_id", _pid(p))
    row["search_level"] = level
    row["level_club"] = club
    row["level_league_code"] = code
    if level == "JUNIOR":
        row.setdefault("league", _league_label(code))
        row.setdefault("team", club)
        row["roster_level"] = row.get("roster_level") or "junior"
    if team is not None:
        row.setdefault("team_id", str(getattr(team, "team_id", None) or getattr(team, "id", "") or ""))
        row.setdefault("team_abbr", _team_label(team))
    undrafted_junior = (
        level == "JUNIOR"
        and team is None
        and not bool(getattr(p, "drafted", False))
        and not (getattr(p, "nhl_rights_team_id", None) or getattr(p, "rights_team_id", None))
    )
    if undrafted_junior:
        # Pre-draft fog: an undrafted prospect's true ability is hidden. The dossier shows
        # his scouting picture and draft ranking, never his OVR, potential or attributes.
        for k in list(row.keys()):
            kl = k.lower()
            if kl in ("ovr", "overall", "effective_ovr", "potential", "potential_score", "pot", "display_ovr",
                      "ratings", "attributes", "attribute_profile", "chapter_profile", "overall_delta",
                      "growth_delta", "season_ovr_delta", "season_potential_delta", "potential_change_season",
                      "rating_history", "true_ovr", "true_potential") or kl.startswith("attr_") or (
                kl != "draft_rank"
                and any(w in kl for w in ("ovr", "overall", "potential", "rating", "attribute"))
            ):
                row.pop(k, None)
        row["scouting_only"] = True
        row["pre_draft"] = True
        rank = None
        try:
            from services.franchise_sim import get_cached_draft_class_rankings

            board = get_cached_draft_class_rankings(session, getattr(session, "sim", None)) or {}
            entries = board.get("entries") or board.get("prospects") or board.get("board") or board.get("rankings") or []
            pid = _pid(p)
            for i, e in enumerate(entries):
                if str(e.get("key") or e.get("player_id") or e.get("id") or "") == pid:
                    rank = int(e.get("rank") or (i + 1))
                    row["draft_projection"] = (
                        e.get("scout_projection") or e.get("projected_range") or e.get("projection") or e.get("draft_range")
                    )
                    row["scout_grade"] = e.get("scout_grade") or e.get("grade")
                    row["draft_board_tier"] = e.get("scout_tier") or e.get("class_tier") or e.get("tier") or e.get("board_tier")
                    row["draft_stock"] = e.get("stock_label") or e.get("draft_stock")
                    break
        except Exception:
            _log.debug("draft rank lookup failed", exc_info=True)
        row["draft_rank"] = rank
    return {"found": True, "player": row}
