"""GM on-ice / development directives set in player meetings.

Shoot More, Shoot Less, Pass More, Pass Less, Play More Physical, Focus Defense and
Develop Playmaking are real levers, not flavour:

* In-game directives are stored on the player (``player.gm_directives``) and merged
  into the per-game franchise stat modifiers (``prime_franchise_game_stat_modifiers``).
  The engine reads them through ``_gm_franchise_alloc_mult``:
  ``shot_selection`` (who shoots inside a deployed five), ``shot_involvement`` (shot
  volume), ``assist_involvement`` (primary/secondary assist share), ``physical_play``
  (hit share) and ``defensive_effort`` / ``defensive_awareness`` (suppression, blocks).
* Development focus biases ``allocate_growth_to_attributes`` toward the matching
  attribute group (playmaking keys for Develop Playmaking).

Whether it sticks depends on the player: coachability, professionalism and trust in the
GM help; ego, volatility and being asked to play against type (a sniper told to shoot
less) hurt. He accepts (full effect), pushes back but tries (half effect), or refuses
(no effect, trust hit).
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

DIRECTIVE_SLOT = {
    "habit_shoot_more": "shot_mix",
    "habit_shoot_less": "shot_mix",
    "habit_pass_more": "shot_mix",
    "habit_pass_less": "shot_mix",
    "habit_play_physical": "physical",
    "habit_focus_defense": "defense",
    "habit_develop_playmaking": "dev_focus",
}

# Engine modifier deltas at full strength.
DIRECTIVE_MODS: Dict[str, Dict[str, float]] = {
    "habit_shoot_more": {"shot_selection": 0.24, "shot_involvement": 0.12, "assist_involvement": -0.06},
    "habit_shoot_less": {"shot_selection": -0.24, "shot_involvement": -0.12, "assist_involvement": 0.08},
    "habit_pass_more": {"assist_involvement": 0.16, "shot_selection": -0.10},
    "habit_pass_less": {"assist_involvement": -0.12, "shot_selection": 0.10},
    "habit_play_physical": {"physical_play": 0.32, "defensive_effort": 0.04},
    "habit_focus_defense": {"defensive_effort": 0.14, "defensive_awareness": 0.05, "shot_involvement": -0.06, "assist_involvement": -0.04},
    "habit_develop_playmaking": {"assist_involvement": 0.04},
}

# Growth-allocation bias: attribute key prefixes -> multiplier.
DEV_FOCUS_BIAS: Dict[str, Dict[str, float]] = {
    "habit_develop_playmaking": {"pm_": 1.85, "iqm_hockey_iq": 1.3, "iqm_game_sense": 1.3},
    "habit_focus_defense": {"def_": 1.3},
    "habit_play_physical": {"phy_": 1.25},
    "habit_shoot_more": {"off_": 1.15},
}

TITLES = {
    "habit_shoot_more": "Tell him to shoot more",
    "habit_shoot_less": "Tell him to shoot less",
    "habit_pass_more": "Ask him to pass more",
    "habit_pass_less": "Tell him to stop passing up shots",
    "habit_play_physical": "Ask him to play more physical",
    "habit_focus_defense": "Ask him to focus on defense",
    "habit_develop_playmaking": "Develop his playmaking",
}

SHORT = {
    "habit_shoot_more": "Shoot More",
    "habit_shoot_less": "Shoot Less",
    "habit_pass_more": "Pass More",
    "habit_pass_less": "Pass Less",
    "habit_play_physical": "Play Physical",
    "habit_focus_defense": "Focus Defense",
    "habit_develop_playmaking": "Develop Playmaking",
}

EFFECT_TEXT = {
    "habit_shoot_more": "He looks for his shot first: more shot attempts, fewer assists.",
    "habit_shoot_less": "He defers to linemates: fewer shots, more set-ups.",
    "habit_pass_more": "He hunts the extra pass: bigger share of assists, fewer shots.",
    "habit_pass_less": "He stops passing up looks: more shots, fewer assists.",
    "habit_play_physical": "He finishes his checks: more hits, a bit more defensive bite.",
    "habit_focus_defense": "He plays a tighter 200-foot game: better suppression and blocks, less offence.",
    "habit_develop_playmaking": "His development reps go into passing, vision and puck distribution.",
}

DIRECTIVE_IDS: Tuple[str, ...] = tuple(DIRECTIVE_SLOT.keys())
HABIT_DAYS = 90


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:12], 16))


def _f(v: Any, d: float = 50.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _avg(player: Any, prefix: str) -> float:
    r = getattr(player, "ratings", None) or {}
    vals = []
    for k, v in r.items():
        if str(k).startswith(prefix):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
    return sum(vals) / len(vals) if vals else 60.0


def player_profile(player: Any) -> Dict[str, float]:
    return {
        "shoot": _avg(player, "off_"),
        "pass": _avg(player, "pm_"),
        "phys": _avg(player, "phy_"),
        "def": _avg(player, "def_"),
    }


def misfit(directive: str, player: Any) -> float:
    """0 = natural fit, 1 = against everything he is."""
    if player is None:
        return 0.3
    pr = player_profile(player)
    gap = pr["shoot"] - pr["pass"]  # + = shooter
    if directive == "habit_shoot_less":
        return max(0.0, min(1.0, (gap + 2) / 10.0))
    if directive == "habit_pass_less":
        return max(0.0, min(1.0, (-gap + 2) / 10.0))
    if directive == "habit_shoot_more":
        return max(0.0, min(0.8, (-gap - 2) / 12.0))
    if directive == "habit_pass_more":
        return max(0.0, min(0.8, (gap - 2) / 12.0))
    if directive == "habit_play_physical":
        return max(0.0, min(1.0, (68.0 - pr["phys"]) / 16.0))
    if directive == "habit_focus_defense":
        return max(0.0, min(0.9, (pr["shoot"] - pr["def"]) / 16.0))
    if directive == "habit_develop_playmaking":
        return max(0.0, min(0.6, gap / 16.0))
    return 0.3


def _is_goalie(ctx: Dict[str, Any]) -> bool:
    return bool(ctx.get("is_goalie"))


def directive_available(ctx: Dict[str, Any], directive: str) -> bool:
    if _is_goalie(ctx):
        return False
    player = ctx.get("player")
    current = active_directives(player)
    slot = DIRECTIVE_SLOT[directive]
    cur = current.get(slot) or {}
    if str(cur.get("id") or "") == directive and cur.get("accepted") in ("accepted", "partial"):
        return False  # already in place
    if directive == "habit_develop_playmaking":
        return float(ctx.get("age") or 26) <= 30
    return bool(ctx.get("on_nhl") or ctx.get("on_ahl"))


# ---------------------------------------------------------------------------
# Meeting templates
# ---------------------------------------------------------------------------

_PLAYER_LINES = {
    "habit_shoot_more": ("You want me shooting more?", "So I've been passing it off too much.", "More shots, huh."),
    "habit_shoot_less": ("You're telling me to shoot less?", "So I'm shooting too much.", "Less shooting. Okay..."),
    "habit_pass_more": ("You want me moving the puck more.", "So I'm holding it too long.", "Pass more, got it — or tell me why."),
    "habit_pass_less": ("You think I'm passing up shots.", "So I should be shooting those.", "I'm too unselfish, is what you're saying."),
    "habit_play_physical": ("You want me hitting more.", "More physical, huh.", "You want me finishing checks."),
    "habit_focus_defense": ("You want me thinking defence first.", "So my own zone needs work.", "Defence first. Okay."),
    "habit_develop_playmaking": ("You want me working on my playmaking.", "So vision and passing, that's the focus.", "Playmaking. I can get behind that — maybe."),
}


def build_templates(ctx: Dict[str, Any], gm_choice: Any) -> List[Dict[str, Any]]:
    """Register-ready template dicts for ``_gm_build_interaction``."""
    out: List[Dict[str, Any]] = []
    for did in DIRECTIVE_IDS:
        short = SHORT[did]
        choices = [
            gm_choice("explain", "Explain it with video and numbers", f"Show him why {short.lower()} helps. Best odds he buys in.",
                      {"profile_changes": {"actor": {"state.focus": 2, "state.gm_trust": 1}}, "relationship": {"communication": 2},
                       "gm_directive": did, "directive_style": "explain"}),
            gm_choice("insist", "Make it a team directive", "Non-negotiable. Full effect if he complies; proud players resent it.",
                      {"profile_changes": {"actor": {"state.focus": 2}}, "relationship": {"respect": 1},
                       "gm_directive": did, "directive_style": "insist"}),
            gm_choice("suggest", "Frame it as a suggestion", "Low pressure. Easier on the relationship, smaller effect.",
                      {"profile_changes": {"actor": {"state.gm_trust": 1}}, "relationship": {"trust": 1},
                       "gm_directive": did, "directive_style": "suggest"}),
        ]
        out.append({
            "id": did,
            "category": "development",
            "title": TITLES[did],
            "player_line": _PLAYER_LINES[did][0],
            "choices": choices,
            "available": directive_available(ctx, did),
            "directive_effect": EFFECT_TEXT[did],
        })
    return out


def directive_opening_pool(mc: Dict[str, Any], itype: str) -> Sequence[str]:
    """His opening when the GM brings a directive. Shaped by how badly it fits him."""
    fit = _f(mc.get("directive_misfit"), 0.3)
    base = list(_PLAYER_LINES.get(itype) or ("Okay. What's this about?",))
    if fit >= 0.6:
        extra = {
            "habit_shoot_less": ("Shooting is literally why I'm in this league.", "My shot's the best thing I've got."),
            "habit_pass_less": ("I make plays. That's my game.", "Setting guys up is what I do best."),
            "habit_play_physical": ("I'm not a banger. You know that, right?", "Running around hitting guys isn't my game."),
            "habit_focus_defense": ("I'm here to score.", "Defence first? That's not why you got me."),
            "habit_shoot_more": ("I'm a pass-first guy. Always have been.",),
            "habit_pass_more": ("I'm a shooter. The puck's supposed to come to me.",),
            "habit_develop_playmaking": ("I'm a finisher, though.",),
        }.get(itype) or ()
        return tuple(f"{a} {b}" for a in base[:2] for b in extra) or tuple(base)
    if fit <= 0.15:
        return tuple(f"{a} Honestly, I've been thinking the same thing." for a in base[:2]) + tuple(base)
    return tuple(base)


def directive_reaction_tail(mc: Dict[str, Any], itype: str, reaction: str, seed: str) -> str:
    warm = reaction in ("opens_up", "receptive")
    pools = {
        "habit_shoot_more": (("I'll put more pucks on net.", "Shoot first. Got it."), ("Fine, I'll fire it. If it's wide, that's on you.",)),
        "habit_shoot_less": (("I'll look for the extra pass.", "Alright, I'll pick my spots."), ("Telling a shooter not to shoot. Okay.",)),
        "habit_pass_more": (("I'll move it quicker.", "I'll find the open guy."), ("So the puck's a hot potato now.",)),
        "habit_pass_less": (("I'll take the look when it's there.", "If I've got it, I'm shooting it."), ("Fine. Don't complain when I miss.",)),
        "habit_play_physical": (("I'll finish my checks.", "Okay. They'll feel me out there."), ("I'll hit people. Don't be surprised when I take penalties.",)),
        "habit_focus_defense": (("I'll take care of my own end first.", "Okay, defence first."), ("So I'm a checker now.",)),
        "habit_develop_playmaking": (("I'll put the reps in on my passing.", "Vision work, every day. Got it."), ("I'll do the drills. Doesn't mean I like it.",)),
    }.get(itype)
    if not pools:
        return ""
    pool = pools[0] if warm else pools[1]
    return pool[_rng(seed, "dt").randrange(len(pool))]


# ---------------------------------------------------------------------------
# Applying the directive
# ---------------------------------------------------------------------------

def active_directives(player: Any) -> Dict[str, Dict[str, Any]]:
    raw = getattr(player, "gm_directives", None) if player is not None else None
    return dict(raw) if isinstance(raw, dict) else {}


def _current_day(session: Any) -> int:
    try:
        return int(getattr(session, "calendar_cursor", 0) or 0)
    except Exception:
        return 0


def acceptance(ctx: Dict[str, Any], directive: str, style: str, scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(ctx.get("personality") or {})
    st = dict(ctx.get("state") or {})
    player = ctx.get("player")
    mf = misfit(directive, player)
    coach = _f(p.get("coachability"), 55)
    pro = _f(p.get("professionalism"), 55)
    ego = _f(p.get("ego"), 50)
    vol = _f(p.get("volatility"), 45)
    trust = _f(st.get("gm_trust"), 55)
    age = _f(ctx.get("age"), 26)
    score = 0.42 * coach + 0.26 * pro + 0.22 * trust - 0.22 * ego - 0.10 * vol - 34.0 * mf + 18.0
    if style == "explain":
        score += 8
    elif style == "insist":
        score += 4 - max(0.0, ego - 60) * 0.35
    elif style == "suggest":
        score += 2
    if age <= 23:
        score += 5  # young players do what they're told
    if age >= 32 and directive in ("habit_play_physical", "habit_develop_playmaking"):
        score -= 6
    reaction = str((scene or {}).get("reaction") or "")
    score += {"opens_up": 10, "receptive": 4, "guarded": -3, "bristles": -10, "boils_over": -22}.get(reaction, 0)
    score += _rng(ctx.get("player_id"), directive, _current_day(ctx.get("session")), "acc").uniform(-6, 6)
    if score >= 56:
        verdict = "accepted"
    elif score >= 40:
        verdict = "partial"
    else:
        verdict = "refused"
    strength = {"accepted": 1.0, "partial": 0.5, "refused": 0.0}[verdict]
    if style == "suggest":
        strength *= 0.7
    return {"verdict": verdict, "strength": round(strength, 2), "score": round(score, 1), "misfit": round(mf, 2)}


def apply_from_meeting(session: Any, ctx: Dict[str, Any], outcome: Dict[str, Any], scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    directive = str(outcome.get("gm_directive") or "")
    if directive not in DIRECTIVE_SLOT:
        return {}
    player = ctx.get("player")
    name = str((ctx.get("entity") or {}).get("player_name") or "He")
    last = name.split()[-1] if name else "He"
    if player is None:
        return {"ok": False, "message": "Directive not applied: player not found."}
    style = str(outcome.get("directive_style") or "explain")
    walked = bool((scene or {}).get("walked_out"))
    res = acceptance(ctx, directive, style, scene)
    if walked:
        res.update({"verdict": "refused", "strength": 0.0})
    day = _current_day(session)
    book = active_directives(player)
    slot = DIRECTIVE_SLOT[directive]
    if res["verdict"] != "refused":
        book[slot] = {
            "id": directive,
            "label": SHORT[directive],
            "strength": res["strength"],
            "accepted": res["verdict"],
            "style": style,
            "set_day": day,
            "until_day": day + (HABIT_DAYS if slot != "dev_focus" else 365),
        }
        try:
            setattr(player, "gm_directives", book)
            cache = getattr(player, "_gm_runtime_cache", None)
            if isinstance(cache, dict):
                cache.clear()
        except Exception:
            pass
    entity = ctx.get("entity") or {}
    state = entity.get("state") if isinstance(entity, dict) else None
    if isinstance(state, dict):
        if res["verdict"] == "refused":
            state["gm_trust"] = max(0.0, _f(state.get("gm_trust"), 55) - 3)
            state["morale"] = max(0.0, _f(state.get("morale"), 55) - 2)
        elif res["verdict"] == "accepted":
            state["focus"] = min(100.0, _f(state.get("focus"), 60) + 2)
    pct = int(round(res["strength"] * 100))
    if res["verdict"] == "accepted":
        msg = f"{last} buys in: “{SHORT[directive]}” is now part of his game ({pct}% effect). {EFFECT_TEXT[directive]}"
    elif res["verdict"] == "partial":
        msg = f"{last} pushes back but says he'll try. “{SHORT[directive]}” applied at {pct}% effect. {EFFECT_TEXT[directive]}"
    else:
        msg = f"{last} refuses to change his game. No directive applied; his trust in you dipped."
    return {
        "ok": True,
        "directive": directive,
        "label": SHORT[directive],
        "verdict": res["verdict"],
        "strength": res["strength"],
        "misfit": res["misfit"],
        "effect": EFFECT_TEXT[directive],
        "message": msg,
        "active": directives_view(player, day),
    }


def directives_view(player: Any, day: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    for slot, row in active_directives(player).items():
        if not isinstance(row, dict):
            continue
        if day is not None and int(row.get("until_day") or 0) < int(day):
            continue
        rows.append({
            "slot": slot,
            "id": row.get("id"),
            "label": row.get("label") or SHORT.get(str(row.get("id")), ""),
            "strength": row.get("strength"),
            "accepted": row.get("accepted"),
            "effect": EFFECT_TEXT.get(str(row.get("id")), ""),
            "until_day": row.get("until_day"),
        })
    return rows


def directive_modifiers(player: Any, day: Optional[int] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in active_directives(player).values():
        if not isinstance(row, dict):
            continue
        if day is not None and int(row.get("until_day") or 0) < int(day):
            continue
        strength = _f(row.get("strength"), 0.0)
        for k, v in (DIRECTIVE_MODS.get(str(row.get("id"))) or {}).items():
            out[k] = out.get(k, 0.0) + float(v) * strength
    return out


def merge_into_game_context(session: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Add directive deltas onto the per-game player modifier rows (both sides)."""
    day = _current_day(session)
    team_by_id = getattr(session, "team_by_id", None) or {}
    for side, tid_key in (("home", "home_team_id"), ("away", "away_team_id")):
        team = team_by_id.get(str(context.get(tid_key) or ""))
        if team is None:
            continue
        block = context.setdefault(side, {})
        mods = dict(block.get("player_modifiers") or {})
        for p in list(getattr(team, "roster", None) or []):
            if not isinstance(getattr(p, "gm_directives", None), dict):
                continue
            d = directive_modifiers(p, day)
            if not d:
                continue
            pid = str(getattr(p, "id", "") or "")
            row = dict(mods.get(pid) or {})
            for k, v in d.items():
                row[k] = float(row.get(k, 0.0) or 0.0) + v
            mods[pid] = row
        block["player_modifiers"] = mods
    return context


def dev_focus_seed_bias(player: Any, seed: Dict[str, float], skill_keys: Sequence[str]) -> Dict[str, float]:
    """Tilt seasonal growth toward the GM's development focus."""
    book = active_directives(player)
    if not book:
        return seed
    out = dict(seed)
    for row in book.values():
        if not isinstance(row, dict):
            continue
        bias = DEV_FOCUS_BIAS.get(str(row.get("id")))
        if not bias:
            continue
        strength = _f(row.get("strength"), 0.0)
        if strength <= 0:
            continue
        base = max((abs(float(v)) for v in out.values()), default=1.0)
        for key in skill_keys:
            k = str(key)
            for pref, mult in bias.items():
                if k == pref or (pref.endswith("_") and k.startswith(pref)):
                    m = 1.0 + (float(mult) - 1.0) * strength
                    cur = abs(float(out.get(k, 0.0)))
                    if cur <= 0 and pref == "pm_" and str(row.get("id")) == "habit_develop_playmaking":
                        cur = base * 0.55  # make sure playmaking keys enter the focus set
                    if cur > 0:
                        out[k] = cur * m
                    break
    return out
