"""CPU clubs calling the user with trade offers (C7).

Every so often a CPU club with a need phones about one of the user's players. The
offer is built by the Trade Finder against that one partner, so it is a deal the
partner's GM really accepts today. Offers sit in ``session.cpu_inbound_trade_offers``
for a few days, show up as a popup and in the Trade Hub, and expire on their own.
"""

from __future__ import annotations

import logging
import random
import zlib
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

OFFER_LIFETIME_DAYS = 4
MAX_ACTIVE_OFFERS = 3
BASE_DAILY_CHANCE = 0.07  # ~ one call every two weeks in the early season
DEADLINE_DAILY_CHANCE = 0.30  # deadline week: the phone rings


def _cursor(session: Any) -> int:
    return int(getattr(session, "calendar_cursor", 0) or getattr(session, "calendar_idx", 0) or 0)


def _ovr(p: Any) -> float:
    for k in ("overall", "ovr", "rating"):
        v = getattr(p, k, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _pos(p: Any) -> str:
    raw = getattr(p, "position", None)
    if raw is None:
        raw = getattr(getattr(p, "identity", None), "position", "")
    return str(getattr(raw, "value", raw) or "")


def active_inbound_offers(session: Any) -> List[Dict[str, Any]]:
    """Live offers, pruned of expired ones and ones whose assets have moved."""
    now = _cursor(session)
    rows = list(getattr(session, "cpu_inbound_trade_offers", None) or [])
    league = getattr(getattr(session, "sim", None), "league", None)
    owners: Dict[str, str] = {}
    for t in list(getattr(league, "teams", None) or []):
        tid = str(getattr(t, "team_id", ""))
        for attr in ("roster", "ahl_roster", "prospect_pool"):
            for p in list(getattr(t, attr, None) or []):
                owners[str(getattr(p, "id", ""))] = tid
    keep: List[Dict[str, Any]] = []
    user_tid = str(getattr(session, "user_team_id", "") or "")
    for o in rows:
        if int(o.get("expires_day", -1)) < now or o.get("status") in ("declined", "accepted"):
            continue
        ok = True
        for a in list(o.get("user_gives") or []):
            if a.get("type") == "player" and owners.get(str(a.get("id"))) != user_tid:
                ok = False
        for a in list(o.get("user_gets") or []):
            if a.get("type") == "player" and owners.get(str(a.get("id"))) != str(o.get("partner_team_id")):
                ok = False
        if ok:
            keep.append(o)
    session.cpu_inbound_trade_offers = keep
    return keep


def decline_inbound_offer(session: Any, offer_id: str) -> Dict[str, Any]:
    rows = list(getattr(session, "cpu_inbound_trade_offers", None) or [])
    for o in rows:
        if str(o.get("offer_id")) == str(offer_id):
            o["status"] = "declined"
    session.cpu_inbound_trade_offers = rows
    return {"ok": True, "offers": active_inbound_offers(session)}


def _pick_target(session: Any, user_team: Any, rng: random.Random, deadline_phase: float) -> Optional[Any]:
    roster = sorted(list(getattr(user_team, "roster", None) or []), key=_ovr, reverse=True)
    if len(roster) < 8:
        return None
    # Clubs call about the middle of the lineup — not the franchise player, not the 23rd man.
    pool = roster[3:18]
    weights = []
    for p in pool:
        w = 1.0
        c = getattr(p, "contract", None)
        yrs = (c.get("years_remaining") if isinstance(c, dict) else getattr(c, "years_remaining", None)) if c is not None else None
        if yrs is not None and int(yrs or 0) <= 1:
            w += 1.5 * deadline_phase  # expiring players are the deadline calls
        if bool(getattr(p, "_trade_demand_active", False)):
            w += 2.0
        weights.append(w)
    if not pool:
        return None
    return rng.choices(pool, weights=weights, k=1)[0]


def maybe_generate_inbound_offer(session: Any, calendar_idx: int) -> Optional[Dict[str, Any]]:
    """Roll for a CPU call today. Cheap when it doesn't fire (one RNG draw)."""
    phase = str(getattr(session, "phase", "") or "").lower()
    if phase not in ("regular", "regular_season", "in_season"):
        return None
    league = getattr(getattr(session, "sim", None), "league", None)
    if league is None:
        return None
    active = active_inbound_offers(session)
    if len(active) >= MAX_ACTIVE_OFFERS:
        return None
    try:
        from app.sim_engine.trades.trade_deadline import days_to_deadline, deadline_phase as _dphase

        max_d = max(40, int(getattr(session, "nhl_regular_season_last_index", 0) or 0))
        days_left = int(days_to_deadline(league, int(calendar_idx), max_d))
        dphase = float(_dphase(league, int(calendar_idx), max_d) or 0.0)
    except Exception:
        days_left, dphase = 99, 0.0
    if days_left < 0:
        return None
    chance = DEADLINE_DAILY_CHANCE if 0 <= days_left <= 7 else BASE_DAILY_CHANCE + 0.10 * dphase
    seed = zlib.crc32(f"inbound|{getattr(session, 'session_id', '')}|{int(calendar_idx)}".encode("utf-8"))
    rng = random.Random(seed)
    if rng.random() >= chance:
        return None

    user_tid = str(getattr(session, "user_team_id", "") or "")
    teams = {str(getattr(t, "team_id", "")): t for t in list(getattr(league, "teams", None) or [])}
    user_team = teams.get(user_tid)
    if user_team is None:
        return None
    from app.sim_engine.trades.trade_value import _pos_group  # noqa: WPS433

    already = {str(o.get("partner_team_id")) for o in active}
    target = None
    scored: List[Any] = []
    for _ in range(4):  # a few names come up on the call sheet before one sticks
        cand = _pick_target(session, user_team, rng, dphase)
        if cand is None:
            return None
        grp = _pos_group(_pos(cand))
        scored = []
        for tid, t in teams.items():
            if tid == user_tid or tid in already:
                continue
            # Who calls: clubs whose own depth at his position is weaker than him.
            same = sorted((_ovr(p) for p in list(getattr(t, "roster", None) or []) if _pos_group(_pos(p)) == grp), reverse=True)
            slots = {"G": 1, "D": 4, "C": 2, "W": 4}.get(grp, 3)
            worst = same[slots - 1] if len(same) >= slots else 0.0
            gap = _ovr(cand) - worst
            if gap > 0:
                scored.append((gap + rng.random() * 2.0, tid))
        if scored:
            target = cand
            break
    if target is None or not scored:
        return None
    scored.sort(reverse=True)
    partner_tid = scored[0][1]
    try:
        from services.trade_finder import find_trade_offers

        res = find_trade_offers(
            session, asset_type="player", asset_id=str(getattr(target, "id", "")), mode="sell",
            target_team_id=partner_tid, limit=1,
        )
    except Exception:
        _log.debug("inbound offer build failed", exc_info=True)
        return None
    offers = list((res or {}).get("offers") or [])
    if not offers:
        return None
    offer = dict(offers[0])
    partner = teams.get(partner_tid)
    pname = str(getattr(target, "name", None) or "one of your players")
    offer.update(
        {
            "offer_id": f"inb_{int(calendar_idx)}_{partner_tid}_{getattr(target, 'id', '')}",
            "inbound": True,
            "created_day": int(calendar_idx),
            "expires_day": int(calendar_idx) + OFFER_LIFETIME_DAYS,
            "status": "open",
            "headline": f"{offer.get('partner_name') or partner_tid} called about {pname}",
        }
    )
    rows = list(getattr(session, "cpu_inbound_trade_offers", None) or [])
    rows.append(offer)
    session.cpu_inbound_trade_offers = rows
    try:
        from services.franchise_sim import _append_showcase_popup

        gets = ", ".join(
            str(a.get("name") or (f"a {a.get('year')} round-{a.get('round')} pick" if a.get("year") else "a pick"))
            for a in list(offer.get("user_gets") or [])[:3]
        )
        _append_showcase_popup(
            session,
            f"inbound_offer:{offer['offer_id']}",
            {
                "kind": "breaking_news",
                "source_label": "Trade Call",
                "headline": offer["headline"],
                "summary": f"They're offering {gets or 'a package'} for {pname}. The offer is open for {OFFER_LIFETIME_DAYS} days — review it in the Trade Hub.",
                "theme": "info",
                "team_abbr": str(getattr(partner, "abbreviation", "") or ""),
            },
        )
    except Exception:
        _log.debug("inbound popup failed", exc_info=True)
    return offer
