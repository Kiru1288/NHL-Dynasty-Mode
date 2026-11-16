"""Retirement talk: a short back-and-forth with a veteran who is weighing retirement.

The GM makes a pitch; the player answers in his own voice and either agrees to come back or
pushes back. Two failed pitches and his mind is made up. Every reply is driven by the save:
his age, why he is thinking about it (the retirement type), his risk, his role, contract and
whether the club looks like a contender.

Decision shape (session.pending_decisions item, kind "retirement_decision"):
  meta.talk:     [{"speaker": "player"|"gm", "text": str}]
  meta.declines: number of pitches he has turned down
  options:       remaining pitches (+ "let_retire")
Finished talks are kept on session.retirement_talks[player_id] for the screen to show.
"""

from __future__ import annotations

import logging
import random
import zlib
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

MAX_DECLINES = 2

PITCHES: Dict[str, Dict[str, str]] = {
    "one_more_year": {
        "label": "Ask for one more year",
        "effect_summary": "He plays one more season on his current deal. Small morale lift.",
        "gm": "I'm not ready to see you go. Give us one more year — this room still needs you.",
    },
    "one_year_deal": {
        "label": "Offer a one-year extension",
        "effect_summary": "Adds a year to his contract (cap cost). Strongest security signal.",
        "gm": "Let's put it in writing. One more year, guaranteed — you finish this on your terms.",
    },
    "reduced_role": {
        "label": "Promise a lighter workload",
        "effect_summary": "Fewer minutes and less wear; he moves down the lineup.",
        "gm": "What if we take the load off? Fewer minutes, rest days when you need them, no back-to-back grind.",
    },
    "leadership_role": {
        "label": "Promise a leadership role",
        "effect_summary": "Letter consideration and a bigger voice in the room.",
        "gm": "The young guys look at you. I want you wearing a letter and running that room.",
    },
    "contender_push": {
        "label": "Promise a run at the Cup",
        "effect_summary": "Creates a win-now promise; he'll be unhappy if the club falls short.",
        "gm": "We're going for it this year. I'll add pieces at the deadline — I want you there when we do.",
    },
}

FAREWELL_GM = "If your mind's made up, I respect it. Thank you for everything you gave this club."


def _seed(*parts: Any) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF


def _pick(options: List[str], *seed_parts: Any) -> str:
    return random.Random(_seed(*seed_parts)).choice(options)


def _first(name: str) -> str:
    return str(name or "").split(" ")[0] or "he"


def _opener(meta: Dict[str, Any]) -> str:
    rtype = str(meta.get("retirement_type") or "")
    age = meta.get("age")
    seasons = meta.get("seasons_played")
    pid = meta.get("player_id")
    if rtype in ("injury_retirement",) or meta.get("recent_major_injury"):
        lines = [
            "My body's telling me something, GM. Mornings are rough, and I don't bounce back like I used to.",
            "I'll be honest — the injury scared me. I don't know if I want to go through another rehab.",
        ]
    elif rtype in ("family_retirement",):
        lines = [
            "My kids are at an age where they notice when I'm gone. I've been thinking about being home.",
            "I've missed a lot of birthdays. I'm starting to think it's time to be there for my family.",
        ]
    elif rtype in ("decline_retirement", "age_retirement", "goalie_longevity_retirement"):
        lines = [
            f"I've been doing this {seasons} seasons. I can feel the game getting quicker than me." if seasons else
            "I can feel the game getting quicker than me. I don't want to hang on too long.",
            f"I'm {age} now. I want to go out while I can still play the way I want to." if age else
            "I want to go out while I can still play the way I want to.",
        ]
    elif rtype in ("role_frustration_retirement",):
        lines = [
            "I didn't come back to sit in the press box. If my role keeps shrinking, I'd rather walk away.",
            "Watching from the fourth line isn't how I want to finish. I'm thinking about hanging them up.",
        ]
    elif rtype in ("morale_retirement",):
        lines = [
            "I'm not having fun anymore, GM. When it stops being fun, it's hard to keep going.",
            "Honestly, the passion's not where it used to be. I don't want to cheat the guys.",
        ]
    elif rtype in ("cup_win_walkaway", "legend_farewell"):
        lines = [
            "I've done about everything I set out to do. Part of me wants to leave on a high note.",
            "I keep thinking there's no better way to go out than this. Maybe it's time.",
        ]
    else:
        lines = [
            "I've been thinking a lot this summer. I'm not sure I've got another season in me.",
            "I wanted you to hear it from me first — I'm thinking about retiring.",
        ]
    return _pick(lines, pid, "open")


def _contender_score(session: Any) -> float:
    """0..1 from last season's standing of the user club (archived season history)."""
    try:
        hist = list(getattr(session, "season_history", None) or [])
        last = hist[-1] if hist and isinstance(hist[-1], dict) else {}
        teams = last.get("teams") or {}
        uid = str(getattr(session, "user_team_id", ""))
        if uid not in teams:
            return 0.5
        ranked = sorted(teams.items(), key=lambda kv: -float((kv[1] or {}).get("pts") or 0))
        rank = next(i for i, (tid, _) in enumerate(ranked) if tid == uid)
        return 1.0 - rank / max(1, len(ranked) - 1)
    except Exception:
        return 0.5


def _accept_chance(session: Any, meta: Dict[str, Any], cid: str) -> float:
    risk = float(meta.get("retirement_risk") or 60)
    rtype = str(meta.get("retirement_type") or "")
    base = max(0.12, min(0.85, 1.05 - risk / 100.0))
    fit = 0.0
    if cid == "one_more_year":
        fit = -0.05
    elif cid == "one_year_deal":
        fit = 0.08 if int(meta.get("contract_years_remaining") or 0) <= 0 else 0.0
        if rtype == "no_contract_retirement":
            fit += 0.25
    elif cid == "reduced_role":
        fit = 0.25 if rtype in ("injury_retirement", "age_retirement", "goalie_longevity_retirement", "decline_retirement") else -0.05
        if rtype == "role_frustration_retirement":
            fit = -0.3
    elif cid == "leadership_role":
        fit = 0.25 if rtype in ("role_frustration_retirement", "morale_retirement") else 0.05
        if meta.get("captaincy_status") in ("C", "captain"):
            fit = -0.05
    elif cid == "contender_push":
        c = _contender_score(session)
        fit = (c - 0.5) * 0.6
        if rtype in ("cup_win_walkaway",):
            fit -= 0.15
    if rtype in ("family_retirement",) and cid != "reduced_role":
        fit -= 0.1
    if int(meta.get("declines") or 0) >= 1:
        fit -= 0.05
    return max(0.05, min(0.92, base + fit))


def _yes_line(cid: str, meta: Dict[str, Any]) -> str:
    pid = meta.get("player_id")
    lines = {
        "one_more_year": [
            "Okay. One more. But I'm holding you to the part about the room needing me.",
            "You know what? I'm not done. One more year — let's make it count.",
        ],
        "one_year_deal": [
            "That means a lot, honestly. Get the paperwork ready — I'm in.",
            "Knowing you want me that much makes it easy. Let's do one more.",
        ],
        "reduced_role": [
            "If I'm not getting run into the ground, yeah, I can do that. I'm back.",
            "That's what I needed to hear. Fewer miles, I can give you a full season.",
        ],
        "leadership_role": [
            "I'd be proud to. The kids deserve someone pushing them. I'm in.",
            "A letter? Yeah... that lights a fire. Count me in for another year.",
        ],
        "contender_push": [
            "If we're really going for it, I'm not missing that. Let's win something.",
            "That's the only reason I'd come back. Don't let me down at the deadline.",
        ],
    }
    return _pick(lines.get(cid, ["Alright. I'll give it one more go."]), pid, cid, "yes")


def _no_line(cid: str, meta: Dict[str, Any], final: bool) -> str:
    pid = meta.get("player_id")
    if final:
        return _pick(
            [
                "I appreciate you fighting for me, I really do. But I've made up my mind. It's time.",
                "You've been great to me. Still, I know in my gut I'm done. I'm retiring.",
            ],
            pid, "final",
        )
    lines = {
        "one_more_year": [
            "It's not about wanting to. It's whether I can still do it at this level.",
            "I hear you, but 'one more year' is what everyone says. I need a real reason.",
        ],
        "one_year_deal": [
            "It was never about the money, GM.",
            "A contract doesn't fix what's going on with me. Thanks, though.",
        ],
        "reduced_role": [
            "Fewer minutes? That's the part I'm already struggling with.",
            "I don't want to be a part-timer. If I play, I want to matter.",
        ],
        "leadership_role": [
            "A letter's nice, but that's not what's keeping me up at night.",
            "The room will be fine without me. That's not it.",
        ],
        "contender_push": [
            "With respect, I've heard 'this is the year' before. I've watched this roster.",
            "I'd love to believe that. I'm just not sure this group is there.",
        ],
    }
    return _pick(lines.get(cid, ["I don't think that changes much for me."]), pid, cid, "no")


def ensure_talk(decision: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fill talk fields on a retirement decision (new or from an older save)."""
    meta = decision.setdefault("meta", {})
    if row:
        for key in (
            "age", "position", "overall", "retirement_type", "retirement_reason", "seasons_played",
            "games_played", "goals", "assists", "points", "goalie_wins", "cups_won", "captaincy_status",
            "contract_years_remaining", "team_abbrev", "legacy_tier",
        ):
            if row.get(key) is not None and meta.get(key) is None:
                meta[key] = row.get(key)
    meta.setdefault("player_id", decision.get("player_id"))
    meta.setdefault("player_name", decision.get("player_name"))
    if not meta.get("talk"):
        meta["talk"] = [{"speaker": "player", "text": _opener(meta)}]
    meta.setdefault("declines", 0)
    meta["max_declines"] = MAX_DECLINES
    name = meta.get("player_name") or decision.get("player_name") or "He"
    decision["title"] = f"{name} is thinking about retiring"
    decision["message"] = meta.get("retirement_reason") or decision.get("summary") or ""
    opts = []
    have = {str((o or {}).get("id") or "") for o in decision.get("options") or [] if isinstance(o, dict)}
    for cid, spec in PITCHES.items():
        if not have or cid in have:
            opts.append({"id": cid, "label": spec["label"], "effect_summary": spec["effect_summary"], "description": spec["effect_summary"]})
    final = int(meta.get("declines") or 0) >= MAX_DECLINES
    if final:
        opts = []
    opts.append({
        "id": "let_retire",
        "label": "Wish him well" if final else "Let him retire",
        "effect_summary": "He retires now; his cap hit comes off the books.",
        "description": "He retires now; his cap hit comes off the books.",
    })
    decision["options"] = opts
    return decision


def _record_talk(session: Any, pid: str, meta: Dict[str, Any], outcome: str) -> None:
    store = getattr(session, "retirement_talks", None)
    if not isinstance(store, dict):
        store = {}
    store[str(pid)] = {
        "player_id": str(pid),
        "player_name": meta.get("player_name"),
        "talk": list(meta.get("talk") or []),
        "outcome": outcome,
        "season": int(getattr(session, "season_calendar_year", 0) or 0),
    }
    session.retirement_talks = store
    # Keep the retirements screen in step with the answer.
    payload = getattr(session, "retirements_payload", None)
    if isinstance(payload, dict):
        for key in ("considering", "team", "all"):
            for r in payload.get(key) or []:
                if isinstance(r, dict) and str(r.get("player_id") or "") == str(pid) and not r.get("confirmed"):
                    r["retirement_status"] = "returning_for_one_more_year" if outcome == "returning" else r.get("retirement_status")
                    r["talk_outcome"] = outcome
        if outcome == "returning":
            payload["considering"] = [
                r for r in payload.get("considering") or [] if not (isinstance(r, dict) and str(r.get("player_id") or "") == str(pid))
            ]


def resolve_retirement_talk(session: Any, decision: Dict[str, Any], choice_id: str) -> Dict[str, Any]:
    """Apply one GM pitch. Called after apply_decision removed the decision from the queue."""
    from services.franchise_retirement import apply_retirement_decision

    ensure_talk(decision)
    meta = decision["meta"]
    pid = str(meta.get("player_id") or decision.get("player_id") or "")
    cid = str(choice_id or "")
    talk: List[Dict[str, str]] = list(meta.get("talk") or [])

    if cid == "let_retire":
        talk.append({"speaker": "gm", "text": FAREWELL_GM})
        talk.append({"speaker": "player", "text": _pick([
            "Thank you. This city, this room... I'll miss it every day.",
            "Thanks for everything. I'll be the loudest guy in the stands.",
        ], pid, "bye")})
        meta["talk"] = talk
        effects = apply_retirement_decision(session, decision, "let_retire") or {}
        _record_talk(session, pid, meta, "retired")
        effects.update({"talk_outcome": "retired"})
        return effects

    spec = PITCHES.get(cid)
    if spec is None:
        raise ValueError(f"Unknown pitch {cid!r}.")
    talk.append({"speaker": "gm", "text": spec["gm"]})
    declines = int(meta.get("declines") or 0)
    chance = _accept_chance(session, meta, cid)
    roll = random.Random(_seed(pid, getattr(session, "season_calendar_year", 0), cid, declines)).random()

    if roll < chance:
        talk.append({"speaker": "player", "text": _yes_line(cid, meta)})
        meta["talk"] = talk
        effects = apply_retirement_decision(session, decision, cid) or {}
        _record_talk(session, pid, meta, "returning")
        effects.update({"talk_outcome": "returning", "pitch": cid})
        return effects

    declines += 1
    final = declines >= MAX_DECLINES
    talk.append({"speaker": "player", "text": _no_line(cid, meta, final)})
    meta["talk"] = talk
    meta["declines"] = declines
    remaining = [o for o in decision.get("options") or [] if isinstance(o, dict) and str(o.get("id")) not in (cid, "let_retire")]
    decision["options"] = remaining
    ensure_talk(decision)
    # Back on the desk, first in line, so the conversation continues.
    pending = list(getattr(session, "pending_decisions", None) or [])
    pending.insert(0, decision)
    session.pending_decisions = pending
    return {"talk_outcome": "final_no" if final else "declined", "pitch": cid}
