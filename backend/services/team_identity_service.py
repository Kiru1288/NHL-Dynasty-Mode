"""Franchise layer for team identities (see SimEngine app/sim_engine/systems/team_identity.py).

* ``refresh_team_identities`` — computes every club's identity, applies season
  momentum, attaches ``team.team_identity`` (read by the SimEngine trade
  evaluator), and caches on the session keyed by season + roster signature.
* ``team_identity_payload`` / ``team_identity_for`` — API shapes for the UI.
* ``identity_fit`` — how well a player suits a club (used by FA + draft hooks).

Identity momentum: at the first refresh of a new season the previous season's
final identity becomes the anchor; in-season roster moves can shift a club only
35% of the way toward what its current roster says.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from services._simengine_bootstrap import ensure_simengine_path

ensure_simengine_path()

from app.sim_engine.systems.team_identity import (  # noqa: E402
    compute_league_identities,
    fit_reason,
    player_identity_fit,
)
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

log = logging.getLogger(__name__)


def _teams(session: Any) -> List[Any]:
    league = getattr(getattr(session, "sim", None), "league", None)
    return list(getattr(league, "teams", None) or [])


def _tid(team: Any) -> str:
    return str(getattr(team, "team_id", "") or getattr(team, "id", "") or "")


def _season(session: Any) -> int:
    try:
        return int(getattr(session, "season_calendar_year", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _roster_signature(session: Any, teams: List[Any]) -> str:
    h = hashlib.sha1()
    h.update(str(_season(session)).encode())
    for t in teams:
        h.update(_tid(t).encode())
        ids = sorted(
            str(getattr(p, "id", "") or getattr(p, "player_id", ""))
            for p in (getattr(t, "roster", None) or [])
            if p is not None
        )
        h.update("|".join(ids).encode())
    return h.hexdigest()


def _draft_identities(session: Any, teams: List[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        from services.draft_team_identity import team_draft_identity
    except Exception:
        return out
    year = _season(session) + 1
    cache: Dict[str, Any] = {}
    for t in teams:
        try:
            out[_tid(t)] = team_draft_identity(session, _tid(t), year, cache=cache)
        except Exception:
            continue
    return out


def refresh_team_identities(session: Any, *, force: bool = False) -> Dict[str, Dict[str, Any]]:
    teams = _teams(session)
    if not teams:
        return {}
    sig = _roster_signature(session, teams)
    cached = getattr(session, "_team_identity_cache", None)
    if not force and isinstance(cached, dict) and cached.get("sig") == sig:
        return cached.get("data") or {}

    state = getattr(session, "team_identity_state", None)
    if not isinstance(state, dict):
        state = {}
    season = _season(session)
    if state.get("season") != season:
        # New season: last season's final read becomes the anchor.
        state = {"season": season, "anchor": dict(state.get("last") or {}), "last": dict(state.get("last") or {})}
    anchor = state.get("anchor") or {}

    try:
        data = compute_league_identities(
            teams,
            prev_scores=anchor,
            draft_identities=_draft_identities(session, teams),
            momentum=0.65 if anchor else 0.0,
        )
    except Exception:
        log.exception("team identity refresh failed")
        return (cached or {}).get("data") or {}

    user_tid = str(getattr(session, "user_team_id", "") or "")
    for t in teams:
        ident = data.get(_tid(t))
        if not ident:
            continue
        ident["is_user"] = _tid(t) == user_tid
        ident["team_name"] = str(getattr(t, "name", "") or "")
        ident["abbreviation"] = str(getattr(t, "abbreviation", None) or getattr(t, "abbrev", "") or "")
        try:
            setattr(t, "team_identity", ident)
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    state["last"] = {tid: dict(v.get("scores") or {}) for tid, v in data.items()}
    session.team_identity_state = state
    session._team_identity_cache = {"sig": sig, "data": data}
    return data


def team_identity_for(session: Any, team_id: Any) -> Optional[Dict[str, Any]]:
    return refresh_team_identities(session).get(str(team_id))


def _public(ident: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "team_id", "team_name", "abbreviation", "is_user", "label", "primary", "secondary", "traits",
        "strength", "profile_pct", "avg_age", "star", "target_labels", "drivers",
    )
    return {k: ident.get(k) for k in keep}


def team_identity_payload(session: Any) -> Dict[str, Any]:
    data = refresh_team_identities(session)
    teams = [_public(v) for v in data.values()]
    teams.sort(key=lambda r: str(r.get("team_name") or ""))
    return {"ok": True, "season": _season(session), "teams": teams}


def identity_fit(team: Any, player: Any) -> float:
    return player_identity_fit(getattr(team, "team_identity", None), player)


def identity_fit_reason(team: Any, player: Any) -> Optional[str]:
    ident = getattr(team, "team_identity", None)
    return fit_reason(ident, player, player_identity_fit(ident, player))


# Draft vocabulary -> team-identity style vocabulary.
DRAFT_STYLE_MAP = {
    "sniper": "sniper",
    "playmaker": "playmaker",
    "power_forward": "power_forward",
    "grinder": "grinder",
    "enforcer": "grinder",
    "two_way": "two_way",
    "balanced": "two_way",
    "scoring_forward": "speedster",
    "offensive_defenseman": "offensive_d",
    "puck_mover": "offensive_d",
    "defensive_defenseman": "defensive_d",
    "shutdown": "defensive_d",
    "two_way_defenseman": "two_way_d",
    "mobile_defenseman": "mobile_d",
}
