"""Negotiation meetings: the human side of contract talks.

Three levers, all of which feed the real contract engine (``evaluate_contract_offer`` /
``compute_player_demand``) instead of being flavour text:

* **Recruiting pitch** (free agents) — sell him on winning, his role, development,
  security or the market. How well it lands depends on what *this* player cares about
  and what your club can honestly offer. One pitch per player per window.
* **Agent meeting** (free agents and your own players) — every player has an agent with a
  style (discreet, leverage, media-savvy, leaker, disruptor). How you approach him moves
  the agent's trust in you (it persists across all his clients) and this deal's tone.
* **Hometown discount ask** (your own players, final contract year) — a real yes/no. If he
  agrees, his next deal with your club is priced below market.

Ledger lives on ``session.negotiation_meetings[window][player_id]``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple
import logging as _logging_swallow
_swallowed_log = _logging_swallow.getLogger(__name__)

PITCHES: Tuple[Tuple[str, str, str], ...] = (
    ("winning", "Sell the chance to win", "We're built to contend and you're a missing piece."),
    ("role", "Sell his role", "Here's exactly where you'd slot in and the minutes you'd get."),
    ("development", "Sell development", "Our staff will make you a better player — here's the plan."),
    ("security", "Sell long-term security", "We want you here for the long haul, not a rental."),
    ("market", "Sell the city & fans", "This market will embrace you — on and off the ice."),
)

AGENT_APPROACHES: Tuple[Tuple[str, str, str], ...] = (
    ("rapport", "Build rapport", "Lunch, no numbers. Invest in the relationship."),
    ("transparent", "Be transparent about the cap", "Show him your cap sheet and what's realistic."),
    ("firm_number", "Put a firm number down early", "Anchor the talks — signals you know his value."),
)


def _uid(session: Any) -> str:
    return str(getattr(session, "user_team_id", "") or "")


def _user_team(session: Any) -> Any:
    return (getattr(session, "team_by_id", None) or {}).get(_uid(session))


def _league(session: Any) -> Any:
    return getattr(getattr(session, "sim", None), "league", None)


def _window(session: Any) -> str:
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    phase = str(getattr(session, "phase", "") or "").lower()
    bucket = "off" if phase in ("offseason", "post_cup") else "season"
    return f"{season}:{bucket}"


def _ledger(session: Any, player_id: str) -> Dict[str, Any]:
    root = getattr(session, "negotiation_meetings", None)
    if not isinstance(root, dict):
        root = {}
        session.negotiation_meetings = root
    win = _window(session)
    # Keep only the current window (old windows are irrelevant and bloat the save).
    for k in [k for k in root.keys() if k != win]:
        root.pop(k, None)
    by_player = root.setdefault(win, {})
    return by_player.setdefault(str(player_id), {})


def _find_player(session: Any, player_id: str) -> Tuple[Any, str]:
    """(player, where) — where is 'own' (user org), 'fa' (free agent) or ''."""
    from services.contract_economy import _player_id

    pid = str(player_id)
    team = _user_team(session)
    if team is not None:
        for attr in ("roster", "ahl_roster", "echl_roster", "injured_reserve"):
            for p in list(getattr(team, attr, None) or []):
                if _player_id(p) == pid:
                    return p, "own"
    league = _league(session)
    for attr in ("free_agents", "overseas_free_agents"):
        for p in list(getattr(league, attr, None) or []):
            if _player_id(p) == pid:
                return p, "fa"
    return None, ""


def _entity(session: Any, player_id: str) -> Dict[str, Any]:
    return dict((getattr(session, "universe_players", None) or {}).get(str(player_id)) or {})


def _traits(session: Any, player: Any) -> Dict[str, float]:
    """0..1 traits: what this player cares about."""
    from services.contract_economy import _player_age, _player_id, _player_negotiation_profile

    prof = _player_negotiation_profile(player)
    ent = _entity(session, _player_id(player))
    pers = dict(ent.get("personality") or {})

    def p100(key: str, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(pers.get(key)) / 100.0))
        except (TypeError, ValueError):
            return fallback

    age = int(_player_age(player) or 27)
    return {
        "winning": p100("competitiveness", prof.get("competitiveness", 0.5)),
        "role": p100("ambition", 0.35 + prof.get("gamble_pref", 0.5) * 0.5),
        "development": max(0.0, min(1.0, (27 - age) / 8.0)),
        "security": max(0.0, min(1.0, prof.get("security_pref", 0.5) * 0.7 + (0.3 if age >= 30 else 0.0))),
        "market": max(0.0, min(1.0, 0.3 + prof.get("loyalty", 0.5) * 0.4)),
        "money": p100("money_focus", 0.35 + prof.get("gamble_pref", 0.5) * 0.4),
        "loyalty": p100("loyalty", prof.get("loyalty", 0.5)),
        "age": float(age),
    }


def _team_strength_rank(session: Any) -> Tuple[int, int]:
    from services.contract_economy import _player_ovr

    league = _league(session)
    teams = list(getattr(league, "teams", None) or [])
    scores = []
    for t in teams:
        ovrs = sorted((float(_player_ovr(p)) for p in list(getattr(t, "roster", None) or [])), reverse=True)[:19]
        scores.append((sum(ovrs) / max(1, len(ovrs)), t))
    scores.sort(key=lambda r: -r[0])
    me = _user_team(session)
    for i, (_, t) in enumerate(scores):
        if t is me:
            return i + 1, len(scores)
    return len(scores) // 2, max(1, len(scores))


def _projected_slot(session: Any, player: Any) -> Tuple[str, float]:
    """Honest depth-chart read for where he'd play on your team, and a 0..1 role score."""
    from services.contract_economy import _player_ovr, _position_bucket

    team = _user_team(session)
    pos = _position_bucket(player)
    ovr = float(_player_ovr(player))
    roster = list(getattr(team, "roster", None) or []) if team is not None else []
    fwd = {"C", "LW", "RW", "F"}
    if pos == "G":
        better = sum(1 for p in roster if _position_bucket(p) == "G" and float(_player_ovr(p)) > ovr)
        return ("Starting goalie", 1.0) if better == 0 else ("Backup goalie", 0.3)
    if pos in fwd:
        better = sum(1 for p in roster if _position_bucket(p) in fwd and float(_player_ovr(p)) > ovr)
        line = better // 3 + 1
        label = {1: "First line", 2: "Second line", 3: "Third line"}.get(line, "Fourth line / depth")
        return label, {1: 1.0, 2: 0.8, 3: 0.45}.get(line, 0.2)
    better = sum(1 for p in roster if _position_bucket(p) not in fwd and _position_bucket(p) != "G" and float(_player_ovr(p)) > ovr)
    pair = better // 2 + 1
    label = {1: "Top pair", 2: "Second pair", 3: "Third pair"}.get(pair, "Depth / press box")
    return label, {1: 1.0, 2: 0.75, 3: 0.4}.get(pair, 0.15)


def _offer_strength(session: Any, player: Any, pitch: str) -> Tuple[float, str]:
    """0..1 how much your club can honestly deliver on a pitch, plus the fact behind it."""
    t = _traits(session, player)
    if pitch == "winning":
        rank, n = _team_strength_rank(session)
        return max(0.0, 1.0 - (rank - 1) / max(1, n - 1)), f"Roster ranks #{rank} of {n} on paper"
    if pitch == "role":
        label, score = _projected_slot(session, player)
        return score, f"Projects as: {label}"
    if pitch == "development":
        return (0.9 if t["age"] <= 24 else 0.5 if t["age"] <= 27 else 0.1), f"Age {int(t['age'])}"
    if pitch == "security":
        return 0.75, "Term is yours to offer"
    if pitch == "market":
        try:
            from services.franchise_sim import _ensure_team_fan_profile

            prof = _ensure_team_fan_profile(session, _uid(session)) or {}
            eng = float(prof.get("engagement") or prof.get("fan_engagement") or 55) / 100.0
        except Exception:
            eng = 0.55
        return max(0.0, min(1.0, eng)), "Fan engagement in your market"
    return 0.5, ""


def _read_priorities(traits: Dict[str, float]) -> List[str]:
    labels = {
        "winning": "wants to win now",
        "role": "wants a bigger role",
        "development": "wants to keep developing",
        "security": "values long-term security",
        "market": "cares about where he lives",
        "money": "is chasing the biggest number",
    }
    keys = sorted(labels.keys(), key=lambda k: -traits.get(k, 0.0))
    return [labels[k] for k in keys[:2]]


def _contract_expiry(player: Any) -> Optional[int]:
    """Expiry year of his current deal (None for an unsigned free agent). A meeting
    concession is tied to the deal it was made under, so it lapses once he re-signs."""
    c = getattr(player, "contract", None)
    val = None
    if isinstance(c, dict):
        val = c.get("expiry_year")
    elif c is not None:
        val = getattr(c, "expiry_year", None)
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _rng(*parts: Any) -> random.Random:
    seed = int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)
    return random.Random(seed)


# --------------------------------------------------------------------------- payloads


def meeting_options(session: Any, player_id: str) -> Dict[str, Any]:
    from services.contract_economy import _contract_years_remaining, _player_name

    player, where = _find_player(session, player_id)
    if player is None:
        return {"ok": False, "reason": "Player not found"}
    led = _ledger(session, player_id)
    traits = _traits(session, player)
    try:
        from app.sim_engine.franchise.player_agent_engine import agent_public_view, get_agent_gm_relationship

        agent = agent_public_view(player, session)
        arel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
        agent["trust"] = round(float(arel.get("agent_gm_trust", 0.55)) * 100)
    except Exception:
        agent = {}
    out: Dict[str, Any] = {
        "ok": True,
        "player_id": str(player_id),
        "player_name": _player_name(player),
        "where": where,
        "priorities": _read_priorities(traits),
        "agent": agent,
        "history": list(led.get("log") or []),
        "interest_bonus": round(_interest_bonus_from_ledger(led), 1),
    }
    if where == "fa":
        pitches = []
        for pid_, label, line in PITCHES:
            strength, fact = _offer_strength(session, player, pid_)
            pitches.append({"id": pid_, "label": label, "line": line, "fact": fact})
        out["pitch"] = {"used": bool(led.get("pitch")), "result": led.get("pitch"), "options": pitches}
    out["agent_meeting"] = {
        "used": bool(led.get("agent")),
        "result": led.get("agent"),
        "options": [{"id": a, "label": l, "line": d} for a, l, d in AGENT_APPROACHES],
    }
    if where == "own":
        yrs = int(_contract_years_remaining(player) or 0)
        disc = getattr(player, "_hometown_discount", None)
        if isinstance(disc, dict) and disc.get("expiry") != _contract_expiry(player):
            disc = None
        out["hometown"] = {
            "eligible": yrs <= 1,
            "reason": "" if yrs <= 1 else f"{yrs} years left on his deal — ask in his final year",
            "used": bool(led.get("hometown")),
            "result": led.get("hometown"),
            "active_discount_pct": (disc or {}).get("pct") if isinstance(disc, dict) and str((disc or {}).get("team_id")) == _uid(session) else None,
            "active_discount_m": (disc or {}).get("amount_m") if isinstance(disc, dict) and str((disc or {}).get("team_id")) == _uid(session) else None,
        }
    return out


def _interest_bonus_from_ledger(led: Dict[str, Any]) -> float:
    total = 0.0
    for key in ("pitch", "agent"):
        row = led.get(key)
        if isinstance(row, dict):
            total += float(row.get("interest_delta") or 0.0)
    return total


# --------------------------------------------------------------------------- actions


def run_meeting(session: Any, player_id: str, kind: str, option: str) -> Dict[str, Any]:
    player, where = _find_player(session, player_id)
    if player is None:
        return {"ok": False, "reason": "Player not found"}
    kind = str(kind or "")
    if kind == "pitch":
        res = _do_pitch(session, player, where, str(option))
    elif kind == "agent":
        res = _do_agent(session, player, where, str(option))
    elif kind == "hometown":
        res = ask_hometown_discount(session, player_id)
    else:
        return {"ok": False, "reason": f"Unknown meeting: {kind}"}
    if res.get("ok"):
        res["options"] = meeting_options(session, player_id)
    return res


def _log(led: Dict[str, Any], text: str) -> None:
    led["log"] = (list(led.get("log") or []) + [text])[-8:]


def _do_pitch(session: Any, player: Any, where: str, pitch: str) -> Dict[str, Any]:
    from services.contract_economy import _player_id, _player_name

    if where != "fa":
        return {"ok": False, "reason": "Recruiting pitches are for free agents."}
    if pitch not in {p[0] for p in PITCHES}:
        return {"ok": False, "reason": "Unknown pitch"}
    led = _ledger(session, _player_id(player))
    if led.get("pitch"):
        return {"ok": False, "reason": "You've already made your pitch to him this window."}
    traits = _traits(session, player)
    care = traits.get(pitch, 0.4)
    strength, fact = _offer_strength(session, player, pitch)
    noise = _rng("pitch", _player_id(player), _window(session), pitch).uniform(-1.5, 1.5)
    # It lands when he cares AND you can back it up. Selling something you can't
    # deliver (a fourth-line role to an ambitious player) backfires.
    # Lands in proportion to how much he cares x how well you can back it up; a pitch
    # you can't deliver on (contending with the 31st roster) actively backfires.
    delta = (care * 1.2) * (strength - 0.45) * 20.0 + (care - 0.4) * 3.0 + noise
    if pitch == "development" and traits["age"] >= 30:
        delta = min(delta, -2.0)
    delta = round(max(-8.0, min(13.0, delta)), 1)
    name = _player_name(player)
    if delta >= 7:
        line = f"{name} leaned in — that's exactly what he wanted to hear."
    elif delta >= 2.5:
        line = f"{name} liked what he heard."
    elif delta > -1:
        line = f"{name} was polite, but it didn't move him much."
    else:
        line = f"That missed. {name} didn't buy it ({fact.lower()})."
    led["pitch"] = {"id": pitch, "interest_delta": delta, "line": line, "fact": fact}
    _log(led, f"Pitch — {dict((p[0], p[1]) for p in PITCHES)[pitch]}: {delta:+.1f} interest")
    return {"ok": True, "kind": "pitch", "interest_delta": delta, "message": line, "fact": fact}


def _do_agent(session: Any, player: Any, where: str, approach: str) -> Dict[str, Any]:
    from services.contract_economy import _player_id

    if approach not in {a[0] for a in AGENT_APPROACHES}:
        return {"ok": False, "reason": "Unknown approach"}
    led = _ledger(session, _player_id(player))
    if led.get("agent"):
        return {"ok": False, "reason": "You've already met his agent this window."}
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent, get_agent_gm_relationship
    except Exception:
        return {"ok": False, "reason": "Agent system unavailable"}
    agent = ensure_player_agent(player, session)
    arel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
    style = str(agent.get("style") or "")
    trust_before = float(arel.get("agent_gm_trust", 0.55))
    rng = _rng("agent", _player_id(player), _window(session), approach)
    # (trust change, interest change, demand shift %) by approach x agent style
    table = {
        "rapport": {"discreet": (0.08, 3, 0.0), "media_savvy": (0.06, 2, 0.0), "leverage": (0.03, 1, 0.0),
                    "leaker": (0.04, 1, 0.0), "disruptor": (0.01, 0, 0.0)},
        "transparent": {"discreet": (0.05, 2, -3.0), "media_savvy": (0.04, 1, -2.0), "leverage": (0.0, 0, -1.0),
                        "leaker": (-0.03, -1, -2.0), "disruptor": (-0.02, -1, 0.0)},
        "firm_number": {"discreet": (-0.02, -1, -1.5), "media_savvy": (0.0, 0, -1.0), "leverage": (0.05, 2, -2.5),
                        "leaker": (0.0, -1, -1.0), "disruptor": (-0.06, -3, 1.5)},
    }
    dt, di, dshift = table[approach].get(style, (0.02, 1, 0.0))
    di = round(di + rng.uniform(-0.8, 0.8), 1)
    # R4: trust is the agent's, not the player's — it moves once per agent per window,
    # however many of his clients you meet through.
    root = getattr(session, "negotiation_meetings", None) or {}
    win_row = root.setdefault(_window(session), {}) if isinstance(root, dict) else {}
    met_agents = set(win_row.get("_agents_met") or [])
    agent_key = str(agent.get("id") or "")
    repeat_agent = agent_key in met_agents
    if repeat_agent:
        dt = 0.0
    else:
        met_agents.add(agent_key)
        if isinstance(win_row, dict):
            win_row["_agents_met"] = sorted(met_agents)
    after = max(0.0, min(1.0, trust_before + dt))
    arel["agent_gm_trust"] = after
    leaked = False
    if approach == "transparent" and style == "leaker" and rng.random() < 0.6:
        leaked = True
        # O3: the leak is real — he uses your numbers as leverage, and the trust you built
        # with him takes a hit.
        dshift = float(dshift) + 2.0
        after = max(0.0, after - 0.03)
        arel["agent_gm_trust"] = after
        try:
            from app.sim_engine.franchise.state import _record_storyline  # noqa: WPS433

            _record_storyline(session, {
                "headline": f"{_player_name_safe(player)}'s camp leaks contract talks",
                "summary": "Numbers from your meeting with his agent are making the rounds.",
                "team_id": _uid(session),
                "player_id": _player_id(player),
                "category": "contract",
                "type": "agent_leak",
                "priority": "MEDIUM",
                "heat": 55,
            })
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    if dshift:
        try:
            setattr(player, "_agent_demand_shift", {"team_id": _uid(session), "pct": float(dshift), "window": _window(session), "expiry": _contract_expiry(player)})
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
    name = str(agent.get("name") or "The agent")
    if repeat_agent:
        line = f"{name} already met you this window. Nothing new on the relationship."
    elif dt >= 0.05:
        line = f"{name} appreciated the approach — the relationship is warmer."
    elif dt > 0:
        line = f"{name} took the meeting. Cordial."
    elif dt == 0 and not repeat_agent:
        line = f"{name} listened and gave nothing away."
    else:
        line = f"{name} didn't like it. Talks just got harder."
    if leaked:
        line += " Your cap numbers have already leaked to the press."
    if dshift < 0:
        line += f" His camp is now framing the ask about {abs(dshift):.1f}% lower."
    elif dshift > 0:
        line += f" He's dug in — the ask went up about {dshift:.1f}%."
    led["agent"] = {
        "id": approach,
        "interest_delta": di,
        "trust_before": round(trust_before * 100),
        "trust_after": round(after * 100),
        "demand_shift_pct": dshift,
        "leaked": leaked,
        "line": line,
    }
    _log(led, f"Agent — {dict((a[0], a[1]) for a in AGENT_APPROACHES)[approach]}: trust {round(trust_before*100)}→{round(after*100)}")
    return {"ok": True, "kind": "agent", "interest_delta": di, "message": line, "agent_trust": round(after * 100)}


def ask_hometown_discount(session: Any, player_id: str) -> Dict[str, Any]:
    """Real ask: he says yes or no. Yes = his next deal with you is priced below market."""
    from services.contract_economy import _contract_years_remaining, _player_name, _player_ovr

    player, where = _find_player(session, player_id)
    if player is None or where != "own":
        return {"ok": False, "reason": "Only your own players can be asked for a hometown discount."}
    if int(_contract_years_remaining(player) or 0) > 1:
        return {"ok": False, "reason": "Ask in the final year of his contract."}
    led = _ledger(session, player_id)
    if led.get("hometown"):
        return {"ok": False, "reason": "You've already asked him this window."}
    traits = _traits(session, player)
    ent = _entity(session, player_id)
    state = dict(ent.get("state") or {})
    gr = dict(ent.get("gm_relationship") or {})
    life = dict(ent.get("life") or {})
    gm_trust = float(state.get("gm_trust", 55) or 55)
    morale = float(state.get("morale", 55) or 55)
    goodwill = float(gr.get("negotiation_goodwill", 55) or 55)
    community = float(life.get("community_connection") or life.get("city_attachment") or 40)
    ovr = float(_player_ovr(player))
    age = traits["age"]
    chance = (
        0.12
        + traits["loyalty"] * 0.34
        + (gm_trust - 55) * 0.006
        + (morale - 55) * 0.004
        + (goodwill - 55) * 0.010
        + (community - 50) * 0.003
        + (0.10 if age >= 31 else 0.0)
        - traits["money"] * 0.22
        - (0.10 if ovr >= 88 else 0.0)
    )
    chance = max(0.04, min(0.85, chance))
    roll = _rng("hometown", player_id, _window(session)).random()
    name = _player_name(player)
    season = int(getattr(session, "season_calendar_year", 0) or 0)
    reasons = []
    if traits["loyalty"] >= 0.65:
        reasons.append("loyal to the club")
    if gm_trust >= 65:
        reasons.append("trusts you")
    if goodwill >= 62:
        reasons.append("goodwill from your meetings")
    if traits["money"] >= 0.6:
        reasons.append("money matters a lot to him")
    if gm_trust <= 45:
        reasons.append("doesn't trust management")
    if roll < chance:
        # Dollar spectrum, $1M-$5M: how loyal he is, how much he likes the city and
        # you, and how big his number is decide where on the range he lands. The
        # discount can never take more than ~45% off his ask (a $1.5M depth deal
        # can't give back $1M), so cheap contracts land below the $1M end.
        from services.contract_economy import compute_player_demand, league_minimum_aav

        team = _user_team(session)
        league = _league(session)
        try:
            base_ask = float(compute_player_demand(player, team, league, context="re_sign").get("want_aav_m") or 0.0)
        except Exception:
            base_ask = 0.0
        size_factor = max(0.0, min(1.0, (base_ask - 3.0) / 9.0))
        spread_roll = _rng("hometown_size", player_id, _window(session)).random()
        spread = (
            0.02
            + traits["loyalty"] * 0.40
            + max(0.0, goodwill - 55) * 0.010
            + max(0.0, community - 50) * 0.004
            + (0.06 if age >= 31 else 0.0)
            + size_factor * 0.45
            + (spread_roll - 0.5) * 0.35
            - traits["money"] * 0.15
        )
        spread = max(0.0, min(1.0, spread))
        amount = 1.0 + 4.0 * spread
        cap = max(0.0, (base_ask - league_minimum_aav(league)) * 0.45) if base_ask > 0 else amount
        amount = round(max(0.05, min(amount, cap)), 2)
        pct = round(100.0 * amount / base_ask, 1) if base_ask > 0 else 0.0
        try:
            setattr(player, "_hometown_discount", {
                "team_id": _uid(session), "amount_m": amount, "pct": pct,
                "base_ask_m": round(base_ask, 3), "season": season, "expiry": _contract_expiry(player),
            })
        except Exception:
            _swallowed_log.debug("suppressed exception", exc_info=True)
        msg = f"{name} agreed — he'll take about ${amount:.2f}M a year under his number to stay."
        led["hometown"] = {"accepted": True, "pct": pct, "amount_m": amount, "chance": round(chance * 100), "line": msg}
    else:
        _bump_entity(session, player_id, {"state.morale": -2.0})
        _bump_gm_ledger(session, player_id, "negotiation_goodwill", -3.0)
        msg = f"{name} turned it down. He wants full market value."
        led["hometown"] = {"accepted": False, "chance": round(chance * 100), "line": msg}
    led["hometown"]["factors"] = reasons
    _log(led, f"Hometown discount ask: {'yes' if led['hometown']['accepted'] else 'no'} ({led['hometown']['chance']}% odds)")
    return {"ok": True, "kind": "hometown", "accepted": led["hometown"]["accepted"], "message": msg,
            "chance_pct": led["hometown"]["chance"], "factors": reasons}


def _player_name_safe(player: Any) -> str:
    try:
        from services.contract_economy import _player_name

        return str(_player_name(player))
    except Exception:
        return "His"


def _bump_entity(session: Any, player_id: str, deltas: Dict[str, float]) -> None:
    ents = getattr(session, "universe_players", None) or {}
    ent = ents.get(str(player_id))
    if not isinstance(ent, dict):
        return
    st = ent.setdefault("state", {})
    for k, d in deltas.items():
        key = k.split(".", 1)[1] if "." in k else k
        st[key] = max(0.0, min(100.0, float(st.get(key, 55) or 55) + float(d)))


def _bump_gm_ledger(session: Any, player_id: str, key: str, delta: float) -> None:
    ents = getattr(session, "universe_players", None) or {}
    ent = ents.get(str(player_id))
    if not isinstance(ent, dict):
        return
    gr = ent.setdefault("gm_relationship", {})
    gr[key] = max(0.0, min(100.0, float(gr.get(key, 55) or 55) + float(delta)))


# --------------------------------------------------------------------------- engine hooks


def interest_adjustment(session: Any, player: Any, team: Any, context: str) -> float:
    """Extra interest (pts on the 0-100 offer scale) for offers from the user's club."""
    from services.contract_economy import _player_id

    if session is None or team is None:
        return 0.0
    tid = str(getattr(team, "team_id", "") or getattr(team, "id", "") or "")
    if tid != _uid(session):
        return 0.0
    root = getattr(session, "negotiation_meetings", None)
    if not isinstance(root, dict):
        led: Dict[str, Any] = {}
    else:
        led = dict((root.get(_window(session)) or {}).get(_player_id(player)) or {})
    bonus = _interest_bonus_from_ledger(led)
    try:
        from app.sim_engine.franchise.player_agent_engine import ensure_player_agent, get_agent_gm_relationship

        agent = ensure_player_agent(player, session)
        arel = get_agent_gm_relationship(session, str(agent.get("id") or ""))
        bonus += (float(arel.get("agent_gm_trust", 0.55)) - 0.55) * 25.0
    except Exception:
        _swallowed_log.debug("suppressed exception", exc_info=True)
    return max(-12.0, min(18.0, bonus))


def demand_multiplier(player: Any, team: Any, context: str) -> float:
    """Price change on his ask for the user's club (hometown discount / agent framing)."""
    tid = str(getattr(team, "team_id", "") or getattr(team, "id", "") or "") if team is not None else ""
    if not tid:
        return 1.0
    mult = 1.0
    own_ctx = str(context or "").lower() not in ("ufa", "free_agent", "fa", "free_agency")
    exp_now = _contract_expiry(player)
    disc = getattr(player, "_hometown_discount", None)
    if (
        own_ctx and isinstance(disc, dict) and str(disc.get("team_id")) == tid
        and disc.get("expiry") == exp_now and not disc.get("amount_m")
    ):
        # Legacy percentage discounts (saves from before the dollar spectrum).
        mult *= 1.0 - float(disc.get("pct") or 0.0) / 100.0
    shift = getattr(player, "_agent_demand_shift", None)
    if isinstance(shift, dict) and str(shift.get("team_id")) == tid and shift.get("expiry") == exp_now:
        mult *= 1.0 + float(shift.get("pct") or 0.0) / 100.0
    # Lowball offers sour talks: each insult adds to his number with your club.
    low = getattr(player, "_lowball_ask_shift", None)
    if isinstance(low, dict) and str(low.get("team_id")) == tid and low.get("expiry") == exp_now:
        mult *= 1.0 + float(low.get("pct") or 0.0) / 100.0
    return max(0.8, min(1.10, mult))


def demand_discount_m(player: Any, team: Any, context: str) -> float:
    """Agreed hometown discount in $M/yr off his ask with this club (own-player talks only)."""
    tid = str(getattr(team, "team_id", "") or getattr(team, "id", "") or "") if team is not None else ""
    if not tid:
        return 0.0
    if str(context or "").lower() in ("ufa", "free_agent", "fa", "free_agency"):
        return 0.0
    disc = getattr(player, "_hometown_discount", None)
    if not isinstance(disc, dict) or str(disc.get("team_id")) != tid:
        return 0.0
    if disc.get("expiry") != _contract_expiry(player):
        return 0.0
    return float(disc.get("amount_m") or 0.0)
