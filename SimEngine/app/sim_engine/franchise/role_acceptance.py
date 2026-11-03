"""Role conversations: getting a player to accept (and embrace) the role he has.

Players don't like being told "fourth line" or "backup". In a meeting the GM can ask
him to accept a defined role: "Accept a bottom-6 role for now", "Accept a third-pair
role", "Embrace the shutdown role", "Power-play specialist", "Accept the backup job".

If he accepts he gets *role acceptance* until ``player._role_accepted_until_day``:

* no morale / role-satisfaction penalty from that usage: ``infer_role_satisfaction_from_deployment``
  (trade_stability_engine) floors his satisfaction at ``player._role_accept_sat_floor``;
* a development boost while in the role, even if he'd prefer a bigger one:
  ``calculate_season_growth_budget`` (progression/development.py) multiplies a positive
  budget by ``player._role_accept_growth_mult``. For depth roles at 24+ the multiplier also
  cancels the sheltered-usage growth cut (x0.88) that the budget applies.

Whether he buys in depends on personality (coachability, professionalism, ego, ambition,
volatility), age, ice time vs what he thinks he has earned (deployment-based
role satisfaction) and trust in the GM. Verdicts: accepted / grudging (half effect) / refused.

Attributes written on the player object (all plain values, pickle-safe):
``_role_accepted_until_day``, ``_role_accepted_season``, ``_role_accept_growth_mult``,
``_role_accept_sat_floor``, ``_role_accept_kind``, ``_role_accept_verdict``.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence

ROLE_PACT_DAYS = 120
DEPTH_GROWTH_CUT = 0.88  # development.py: fourth/depth/scratch role at 24+ -> mod *= 0.88

ROLE_IDS = (
    "rolepact_bottom_six",
    "rolepact_third_pair",
    "rolepact_shutdown",
    "rolepact_pp_specialist",
    "rolepact_backup_goalie",
)

TITLES = {
    "rolepact_bottom_six": "Accept a bottom-6 role for now",
    "rolepact_third_pair": "Accept a third-pair role for now",
    "rolepact_shutdown": "Embrace the shutdown role",
    "rolepact_pp_specialist": "Become our power-play specialist",
    "rolepact_backup_goalie": "Accept the backup job",
}

SHORT = {
    "rolepact_bottom_six": "Bottom-6 for now",
    "rolepact_third_pair": "Third pair for now",
    "rolepact_shutdown": "Shutdown role",
    "rolepact_pp_specialist": "PP specialist",
    "rolepact_backup_goalie": "Backup goalie",
}

EFFECT_TEXT = {
    "rolepact_bottom_six": "No sulking over depth minutes; his development keeps moving in a sheltered role.",
    "rolepact_third_pair": "Content on the third pair; growth reps still count.",
    "rolepact_shutdown": "Takes pride in killing plays: no role complaints, and his development keeps moving.",
    "rolepact_pp_specialist": "Lives for the power play: happy with lighter even-strength minutes.",
    "rolepact_backup_goalie": "Supports the starter, stays sharp in practice, no workload grumbling.",
}

DEPTH_ROLES = ("rolepact_bottom_six", "rolepact_third_pair", "rolepact_backup_goalie")

_PLAYER_LINES = {
    "rolepact_bottom_six": ("So I'm a bottom-six guy now.", "You want me to be okay with third and fourth line minutes.", "Bottom six. For now, you said?"),
    "rolepact_third_pair": ("Third pair. That's where you see me?", "So I'm on the third pair for now.", "You want me fine with third-pair minutes."),
    "rolepact_shutdown": ("A shutdown guy, huh.", "So my job is stopping the other team's best.", "You want me as the matchup guy."),
    "rolepact_pp_specialist": ("Power-play guy. That's the pitch?", "So the PP is my thing now.", "You want me living on the power play."),
    "rolepact_backup_goalie": ("So I'm the backup.", "You want me okay with fewer starts.", "Backup. That's where we are?"),
}


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:12], 16))


def _f(v: Any, d: float = 50.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _day(session: Any) -> int:
    try:
        return int(getattr(session, "calendar_cursor", 0) or 0)
    except Exception:
        return 0


def _season(session: Any) -> int:
    try:
        return int(getattr(session, "season_calendar_year", 0) or 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Reading the flag (used by trade_stability_engine and progression)
# ---------------------------------------------------------------------------

def role_pact_active(player: Any, session: Any = None) -> bool:
    if player is None:
        return False
    until = getattr(player, "_role_accepted_until_day", None)
    if until is None:
        return False
    if session is None:
        return True
    try:
        if int(getattr(player, "_role_accepted_season", 0) or 0) not in (0, _season(session)):
            return False
        return int(until) >= _day(session)
    except (TypeError, ValueError):
        return False


def role_satisfaction_floor(player: Any, session: Any = None) -> Optional[float]:
    """Floor on deployment-based role satisfaction while he has accepted his role."""
    if not role_pact_active(player, session):
        return None
    floor = getattr(player, "_role_accept_sat_floor", None)
    return _f(floor, 0.0) if floor is not None else None


def apply_role_satisfaction_floor(player: Any, session: Any, value: Optional[float]) -> Optional[float]:
    floor = role_satisfaction_floor(player, session)
    if floor is None or value is None:
        return value
    return max(float(value), float(floor))


def clear_role_pact(player: Any) -> None:
    for k in ("_role_accepted_until_day", "_role_accepted_season", "_role_accept_growth_mult",
              "_role_accept_sat_floor", "_role_accept_kind", "_role_accept_verdict"):
        try:
            if hasattr(player, k):
                delattr(player, k)
        except Exception:
            try:
                setattr(player, k, None)
            except Exception:
                pass


def sweep_role_pacts(session: Any) -> int:
    """Expire finished pacts (called per game day from the stat bridge). Cheap: only the
    small ``session.role_pact_registry`` is scanned; players are looked up only on expiry."""
    reg = getattr(session, "role_pact_registry", None)
    if not isinstance(reg, dict) or not reg:
        return 0
    day, season = _day(session), _season(session)
    expired = [pid for pid, row in reg.items()
               if not isinstance(row, dict) or int(row.get("season") or 0) != season or int(row.get("until_day") or 0) < day]
    if not expired:
        return 0
    try:
        from app.sim_engine.franchise.storyline_engine import _player_from_roster  # noqa: WPS433
    except Exception:
        _player_from_roster = None  # type: ignore
    for pid in expired:
        reg.pop(pid, None)
        p = _player_from_roster(session, pid) if _player_from_roster else None
        if p is not None:
            clear_role_pact(p)
    session.role_pact_registry = reg
    return len(expired)


# ---------------------------------------------------------------------------
# Meeting templates
# ---------------------------------------------------------------------------

def _is_d(ctx: Dict[str, Any]) -> bool:
    return str(ctx.get("position") or "").upper() in ("D", "LD", "RD", "DEF", "DEFENSE")


def role_available(ctx: Dict[str, Any], rid: str) -> bool:
    if not ctx.get("on_nhl"):
        return False
    player = ctx.get("player")
    if role_pact_active(player, ctx.get("session")) and str(getattr(player, "_role_accept_kind", "") or "") == rid:
        return False
    goalie = bool(ctx.get("is_goalie"))
    rank = int(ctx.get("line_rank") or 0)
    sat = _f((ctx.get("state") or {}).get("role_satisfaction"), 60)
    scratched = bool(ctx.get("scratched"))
    if rid == "rolepact_backup_goalie":
        return goalie and sat <= 70
    if goalie:
        return False
    if rid == "rolepact_bottom_six":
        return not _is_d(ctx) and (rank in (3, 4) or scratched or sat <= 60)
    if rid == "rolepact_third_pair":
        return _is_d(ctx) and (rank >= 3 or scratched or sat <= 60)
    if rid == "rolepact_shutdown":
        return int(ctx.get("pk_unit") or 0) > 0 or _is_d(ctx) or rank >= 3
    if rid == "rolepact_pp_specialist":
        return bool(ctx.get("on_pp")) and (rank >= 3 or sat <= 64)
    return False


def build_templates(ctx: Dict[str, Any], gm_choice: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rid in ROLE_IDS:
        choices = [
            gm_choice("path", "Show him the path back up", "It's a role for now, with a clear way to earn more. Best for ambitious players.",
                      {"profile_changes": {"actor": {"state.gm_trust": 1}}, "relationship": {"communication": 2},
                       "gm_role_pact": rid, "role_style": "path"}),
            gm_choice("honest", "Be straight: this is the job", "Honest and clear. Pros respect it; big egos don't.",
                      {"profile_changes": {"actor": {"state.focus": 2}}, "relationship": {"respect": 2},
                       "gm_role_pact": rid, "role_style": "honest"}),
            gm_choice("value", "Sell how much the team needs it", "Make the role matter. Works on team-first guys and veterans.",
                      {"profile_changes": {"actor": {"state.belonging": 2}}, "relationship": {"loyalty": 1},
                       "gm_role_pact": rid, "role_style": "value"}),
        ]
        out.append({
            "id": rid,
            "category": "role",
            "title": TITLES[rid],
            "player_line": _PLAYER_LINES[rid][0],
            "choices": choices,
            "available": role_available(ctx, rid),
            "role_effect": EFFECT_TEXT[rid],
        })
    return out


def role_opening_pool(mc: Dict[str, Any], itype: str) -> Sequence[str]:
    base = list(_PLAYER_LINES.get(itype) or ("Okay. Where do I fit?",))
    sat = _f(mc.get("role_satisfaction"), 60)
    if sat <= 45:
        return tuple(f"{b} I think I've earned more than that." for b in base[:2]) + tuple(base[:1])
    if sat >= 70:
        return tuple(f"{b} Honestly, I'm fine with how I'm being used." for b in base[:2]) + tuple(base)
    return tuple(base)


def role_reaction_tail(mc: Dict[str, Any], itype: str, reaction: str, seed: str) -> str:
    warm = reaction in ("opens_up", "receptive")
    pools = {
        "rolepact_bottom_six": (("I'll make those minutes count.", "Okay. I'll be the guy you can trust on the third line."), ("For now. I'm holding you to that.",)),
        "rolepact_third_pair": (("I'll keep it simple and earn my way up.", "Third pair, okay. I'll be steady."), ("I'll play it. Doesn't mean I like it.",)),
        "rolepact_shutdown": (("Give me their top line. I'll handle it.", "Shutdown guy. I can own that."), ("So no points for me this year. Great.",)),
        "rolepact_pp_specialist": (("Get me the puck on the half-wall.", "The power play's mine then."), ("PP minutes don't make up for the rest.",)),
        "rolepact_backup_goalie": (("I'll be ready when you need me.", "I'll push him every day in practice."), ("I'll sit. I won't pretend I'm happy.",)),
    }.get(itype)
    if not pools:
        return ""
    pool = pools[0] if warm else pools[1]
    return pool[_rng(seed, "rt").randrange(len(pool))]


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def acceptance(ctx: Dict[str, Any], rid: str, style: str, scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(ctx.get("personality") or {})
    st = dict(ctx.get("state") or {})
    coach = _f(p.get("coachability"), 55)
    pro = _f(p.get("professionalism"), 55)
    ego = _f(p.get("ego"), 50)
    amb = _f(p.get("ambition"), 55)
    vol = _f(p.get("volatility"), 45)
    team_first = _f(p.get("loyalty", p.get("team_first")), 55)
    trust = _f(st.get("gm_trust"), 55)
    sat = _f(st.get("role_satisfaction"), 60)  # deployment-based: ice time vs expectation
    age = _f(ctx.get("age"), 26)
    score = (0.32 * coach + 0.22 * pro + 0.24 * trust + 0.10 * team_first
             - 0.20 * ego - 0.12 * max(0.0, amb - 50) - 0.10 * vol + 12.0)
    # Ice time vs expectation: the further below what he thinks he's earned, the harder.
    score += (sat - 55.0) * 0.38
    if age <= 22:
        score += 6 if rid in DEPTH_ROLES else 3  # "for now" lands with kids
    elif 24 <= age <= 29 and rid in DEPTH_ROLES:
        score -= 5  # prime-age players want the bigger chair
    elif age >= 31:
        score += 7 if rid in ("rolepact_shutdown", "rolepact_backup_goalie", "rolepact_bottom_six") else 2
    if style == "path":
        score += 4 + max(0.0, amb - 60) * 0.25
    elif style == "honest":
        score += 5 - max(0.0, ego - 62) * 0.35 + max(0.0, pro - 60) * 0.15
    elif style == "value":
        score += 3 + max(0.0, team_first - 55) * 0.2 + (4 if age >= 30 else 0)
    reaction = str((scene or {}).get("reaction") or "")
    score += {"opens_up": 10, "receptive": 4, "guarded": -3, "bristles": -10, "boils_over": -22}.get(reaction, 0)
    score += _rng(ctx.get("player_id"), rid, _day(ctx.get("session")), "racc").uniform(-6, 6)
    verdict = "accepted" if score >= 52 else "grudging" if score >= 38 else "refused"
    return {"verdict": verdict, "score": round(score, 1), "role_satisfaction": round(sat, 1)}


def _growth_mult(rid: str, verdict: str, age: float) -> float:
    boost = {"accepted": 1.10, "grudging": 1.04}.get(verdict, 1.0)
    if rid in DEPTH_ROLES and age >= 24:
        boost /= DEPTH_GROWTH_CUT  # cancel the sheltered-role cut while he's bought in
    return round(boost, 3)


def apply_from_meeting(session: Any, ctx: Dict[str, Any], outcome: Dict[str, Any], scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rid = str(outcome.get("gm_role_pact") or "")
    if rid not in TITLES:
        return {}
    player = ctx.get("player")
    name = str((ctx.get("entity") or {}).get("player_name") or "He")
    last = name.split()[-1] if name else "He"
    if player is None:
        return {"ok": False, "message": "Role talk not applied: player not found."}
    style = str(outcome.get("role_style") or "path")
    res = acceptance(ctx, rid, style, scene)
    if (scene or {}).get("walked_out"):
        res["verdict"] = "refused"
    day, season = _day(session), _season(session)
    age = _f(ctx.get("age"), 26)
    entity = ctx.get("entity") or {}
    state = entity.get("state") if isinstance(entity, dict) else None
    if res["verdict"] != "refused":
        mult = _growth_mult(rid, res["verdict"], age)
        floor = 68.0 if res["verdict"] == "accepted" else 58.0
        until = day + ROLE_PACT_DAYS
        try:
            player._role_accepted_until_day = until
            player._role_accepted_season = season
            player._role_accept_growth_mult = mult
            player._role_accept_sat_floor = floor
            player._role_accept_kind = rid
            player._role_accept_verdict = res["verdict"]
            cache = getattr(player, "_gm_runtime_cache", None)
            if isinstance(cache, dict):
                cache.clear()
        except Exception:
            pass
        reg = getattr(session, "role_pact_registry", None)
        reg = dict(reg) if isinstance(reg, dict) else {}
        reg[str(ctx.get("player_id") or getattr(player, "id", ""))] = {"until_day": until, "season": season, "kind": rid}
        try:
            session.role_pact_registry = reg
        except Exception:
            pass
        if isinstance(state, dict):
            state["role_satisfaction"] = max(_f(state.get("role_satisfaction"), 55), floor)
            state["morale"] = min(100.0, _f(state.get("morale"), 55) + (3 if res["verdict"] == "accepted" else 1))
        pct = 10 if res["verdict"] == "accepted" else 4  # net boost; the rest only cancels the depth-role cut
        if res["verdict"] == "accepted":
            msg = f"{last} accepts the role: “{SHORT[rid]}”. No more role complaints, no growth penalty from the role, and +{pct}% development while he's in it. {EFFECT_TEXT[rid]}"
        else:
            msg = f"{last} grudgingly goes along with “{SHORT[rid]}”. Role complaints ease and development gets +{pct}% while it holds."
    else:
        if isinstance(state, dict):
            state["gm_trust"] = max(0.0, _f(state.get("gm_trust"), 55) - 3)
            state["morale"] = max(0.0, _f(state.get("morale"), 55) - 2)
            state["role_satisfaction"] = max(0.0, _f(state.get("role_satisfaction"), 55) - 3)
        mult = 1.0
        msg = f"{last} won't accept “{SHORT[rid]}”. He still thinks he's earned more, and his trust in you dipped."
    return {
        "ok": True,
        "role_pact": rid,
        "label": SHORT[rid],
        "verdict": res["verdict"],
        "score": res["score"],
        "growth_mult": mult,
        "effect": EFFECT_TEXT[rid],
        "message": msg,
        "active": role_pact_view(player, session),
    }


def role_pact_view(player: Any, session: Any = None) -> List[Dict[str, Any]]:
    """Rows in the same shape as player_directives.directives_view."""
    if not role_pact_active(player, session):
        return []
    rid = str(getattr(player, "_role_accept_kind", "") or "")
    verdict = str(getattr(player, "_role_accept_verdict", "accepted") or "accepted")
    mult = _f(getattr(player, "_role_accept_growth_mult", 1.0), 1.0)
    return [{
        "slot": "role_pact",
        "id": rid,
        "label": f"Role: {SHORT.get(rid, rid)}",
        "strength": 1.0 if verdict == "accepted" else 0.5,
        "accepted": "accepted" if verdict == "accepted" else "partial",
        "effect": f"{EFFECT_TEXT.get(rid, '')} Development x{mult:.2f}.",
        "until_day": getattr(player, "_role_accepted_until_day", None),
    }]
