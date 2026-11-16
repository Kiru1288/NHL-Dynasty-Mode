"""Development and potential goals set in GM meetings.

Development goal  one skill area (skating, shooting, ...) and a target in rating points over a
                  few weeks. Each sim day he grows that area for real (ratings change), at a
                  pace set by his age, work ethic, coachability, how he took the meeting and
                  how close he is to his ceiling. At the deadline it's hit / partial / missed.
Potential goal    a season-long target for his overall. Growth spreads across his whole game.
                  Hitting it raises his potential (ceiling), plus the reward he asked for.

Hits on stretch or ambitious goals raise his potential a little; misses on big goals cost
morale and trust. All state lives on session.development_goals[player_id].
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Skill areas -> real rating keys (Player.ratings, 0..1 scale)
# ---------------------------------------------------------------------------

FOCUS: Dict[str, Dict[str, Any]] = {
    "skating": {"label": "Skating", "keys": ("skg_speed", "skg_acceleration", "skg_agility", "skg_edge_work", "skg_explosiveness", "skg_transition_speed"), "pos": "skater"},
    "shooting": {"label": "Shooting", "keys": ("off_wrist_shot_accuracy", "off_wrist_shot_power", "off_finishing", "off_one_timer", "off_shot_iq", "off_puck_placement"), "pos": "skater"},
    "playmaking": {"label": "Playmaking", "keys": ("pm_passing_accuracy", "pm_passing_vision", "pm_playmaking_creativity", "pm_offensive_read", "pm_decision_making", "pm_puck_distribution"), "pos": "skater"},
    "puck_skills": {"label": "Puck skills", "keys": ("pc_puck_control", "pc_stickhandling", "pc_deking", "pc_puck_protection", "pc_control_under_pressure"), "pos": "skater"},
    "defence": {"label": "Defensive game", "keys": ("def_defensive_awareness", "def_gap_control", "def_stick_checking", "def_defensive_reads", "def_body_positioning", "def_interception_skill"), "pos": "skater"},
    "physical": {"label": "Strength and battles", "keys": ("phy_strength", "phy_checking", "phy_balance", "def_board_battles", "phy_physicality"), "pos": "skater"},
    "hockey_iq": {"label": "Hockey sense", "keys": ("iqm_hockey_iq", "iqm_game_sense", "iqm_awareness", "pm_offensive_anticipation", "def_defensive_iq"), "pos": "skater"},
    "conditioning": {"label": "Conditioning", "keys": ("phy_endurance", "phy_stamina", "phy_recovery_rate"), "pos": "any"},
    "faceoffs": {"label": "Faceoffs", "keys": ("def_faceoffs",), "pos": "C"},
    "consistency": {"label": "Consistency", "keys": ("iqm_consistency", "iqm_focus", "iqm_composure", "def_backchecking_effort"), "pos": "skater"},
    "g_positioning": {"label": "Positioning", "keys": ("g_positioning",), "pos": "G"},
    "g_rebounds": {"label": "Rebound control", "keys": ("g_rebound_control_g",), "pos": "G"},
    "g_reflexes": {"label": "Reflexes and athleticism", "keys": ("g_reflexes", "g_athleticism"), "pos": "G"},
    "g_mental": {"label": "Mental game", "keys": ("iqm_composure", "iqm_focus", "iqm_consistency"), "pos": "G"},
}

# Targets in rating points (0-99) and days.
LEVELS: Dict[str, Dict[str, Any]] = {
    "steady": {"label": "Steady", "dev": (2.0, 28), "pot": (2.0, 70), "pace": 1.0, "potential_bonus": 0.0},
    "stretch": {"label": "Stretch", "dev": (4.0, 35), "pot": (4.0, 70), "pace": 1.02, "potential_bonus": 1.0},
    "ambitious": {"label": "Ambitious", "dev": (6.0, 42), "pot": (6.0, 70), "pace": 0.94, "potential_bonus": 2.0},
}

BUY_IN = {"opens_up": 1.12, "receptive": 1.0, "guarded": 0.82, "bristles": 0.62, "boils_over": 0.45}
MAX_DEV_GOALS_PER_SEASON = 3

# What he says when you pick the area (warm = he agrees it's the right thing, cold = he doesn't).
FOCUS_LINES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = {
    # (GM says, his warm reply, his cold reply)
    "skating": (("Your skating. A step quicker and everything else opens up.",),
                ("Yeah. I get caught flat-footed too much. Let's fix it.", "Skating. I've known that for a while, honestly."),
                ("My skating? I thought my feet were fine.", "Skating. Okay, if that's what you see.")),
    "shooting": (("Your shot. Quicker release, better spots. You should be scoring more.",),
                 ("I want to score more. I'm in.", "My release. Yeah. I'll shoot a few hundred a day."),
                 ("I thought my shot was a strength.", "Shooting. Okay.")),
    "playmaking": (("Your playmaking. See the ice, move it quicker, make the guys around you better.",),
                   ("I like that. I see plays I don't make yet.", "Okay. Vision and passing. I can get behind that."),
                   ("I'm a shooter. That's my game.", "Playmaking. Okay, I guess.")),
    "puck_skills": (("Your hands. Protect the puck, beat guys one-on-one.",),
                    ("Yeah. I lose too many pucks in traffic.", "Okay. I'll be on the ice early with a bucket of pucks."),
                    ("My hands are fine.", "Okay. Stickhandling.")),
    "defence": (("Your own-zone game. Gaps, reads, sticks. That's what keeps you on the ice late in games.",),
                ("Fair. I get burned more than I should.", "Okay. If it gets me more minutes, I'm all in."),
                ("Defence. Not exactly what I was hoping for.", "So you don't trust me in my own end.")),
    "physical": (("Strength. Win more battles on the wall and in front of the net.",),
                 ("Yeah. I get pushed off pucks. I'll live in the gym.", "Okay. More muscle. I'm in."),
                 ("I'm strong enough.", "Okay. More lifting.")),
    "hockey_iq": (("Your reads. Play faster in your head and the rest follows.",),
                  ("Yeah. I can feel when I'm a step late. I'll watch more tape.", "Okay. That makes sense."),
                  ("That's a tough one to just work on.", "Hockey sense. Okay.")),
    "conditioning": (("Your conditioning. You fade late in games, and the third period is when we need you.",),
                     ("Fair. I'm gassed by the third. I'll fix it.", "Okay. I'll be the fittest guy in camp."),
                     ("I'm in good shape.", "Okay. More bike.")),
    "faceoffs": (("Faceoffs. Win the draw and we start with the puck.",),
                 ("Yeah. That's a quick way to earn trust. A hundred a day.", "Okay. Draws."),
                 ("Faceoffs? That's it?", "Okay.")),
    "consistency": (("Consistency. Same player every night, not just the good nights.",),
                    ("Yeah. I know I'm streaky. Let's fix that.", "Fair. I want to be reliable."),
                    ("I'm consistent.", "Okay.")),
    "g_positioning": (("Your positioning. Be square and let the puck hit you.",),
                      ("Yeah. I've been scrambling too much.", "Okay. Square and calm. I can do that."),
                      ("My positioning's not the issue.", "Okay.")),
    "g_rebounds": (("Rebound control. Kill the second chances.",),
                   ("Fair. Too many rebounds lately.", "Okay. I'll work on swallowing pucks."),
                   ("My rebounds are fine.", "Okay.")),
    "g_reflexes": (("Your reflexes and athleticism. Get to the pucks you can't see.",),
                   ("Yeah. I'll work with the goalie coach on it.", "Okay. Quicker side to side."),
                   ("I'm quick enough.", "Okay.")),
    "g_mental": (("Your mental game. Bad goals can't turn into bad periods.",),
                 ("Yeah. One goal gets in my head sometimes.", "Fair. I'll work on resetting."),
                 ("My head's fine.", "Okay.")),
}

DEV_ASK = ("How big a jump are we talking? And by when?", "What's the target? I want a number to chase.")
POT_ASK = ("If I get there, what's in it for me?", "And if I hit it? What changes?")

POT_REWARDS: List[Tuple[str, str, str, str]] = [
    # (id, button, GM says, his goodbye)
    ("role", "A bigger role when you hit it", "You hit it, you move up the lineup. Simple as that.", "Hit it and I move up. Okay. Watch me."),
    ("contract", "A contract conversation", "Get there and we sit down about your next contract.", "A contract talk. That's motivation. I'm in."),
    ("growth", "The growth is the reward", "The growth is the reward. You'll be a better player, and that pays for itself.", "Fair enough. I'll do it for me."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _day(session: Any) -> int:
    return int(_f(getattr(session, "calendar_cursor", 0), 0))


def _season(session: Any) -> int:
    return int(_f(getattr(session, "season_calendar_year", 0), 0))


def _pos(player: Any) -> str:
    raw = getattr(player, "position", "") or ""
    return str(getattr(raw, "value", raw) or "").upper()


def _ratings(player: Any) -> Dict[str, Any]:
    r = getattr(player, "ratings", None)
    return r if isinstance(r, dict) else {}


def eligible_focus(player: Any) -> List[str]:
    pos = _pos(player)
    goalie = pos.startswith("G")
    out = []
    for fid, spec in FOCUS.items():
        p = spec["pos"]
        if goalie and p not in ("G", "any"):
            continue
        if not goalie and p == "G":
            continue
        if p == "C" and pos != "C":
            continue
        if all(k in _ratings(player) for k in spec["keys"]):
            out.append(fid)
    return out


def area_score(player: Any, fid: str) -> float:
    """0-99 average of an area's ratings."""
    r = _ratings(player)
    keys = [k for k in FOCUS[fid]["keys"] if k in r]
    if not keys:
        return 0.0
    vals = [_f(r[k]) for k in keys]
    return round(sum(v * 100.0 if v <= 1.5 else v for v in vals) / len(vals), 1)


def suggested_focus(player: Any, n: int = 4) -> List[str]:
    """Areas with the most room to grow (lowest), keeping faceoffs for centres."""
    opts = eligible_focus(player)
    return sorted(opts, key=lambda f: area_score(player, f))[:n]


def _ovr99(player: Any) -> float:
    try:
        inv = getattr(player, "_invalidate_ovr_memo", None)
        if callable(inv):
            inv()
        v = float(player.ovr())
        return v * 100.0 if v <= 1.5 else v
    except Exception:
        return 0.0


def _ceiling99(player: Any) -> float:
    r = _ratings(player)
    for k in ("dev_ceiling", "dev_potential"):
        if r.get(k) is not None:
            v = _f(r[k])
            return v * 100.0 if v <= 1.5 else v
    v = _f(getattr(player, "potential", 0.75), 0.75)
    return v * 100.0 if v <= 1.5 else v


def _age(player: Any) -> int:
    return int(_f(getattr(player, "age", 25), 25))


def _capacity(player: Any, buy_in: float, level: str) -> float:
    age = _age(player)
    a = 1.25 if age <= 21 else 1.1 if age <= 24 else 0.95 if age <= 27 else 0.75 if age <= 30 else 0.5
    r = _ratings(player)
    we = _f(r.get("dev_work_ethic"), 0.6)
    we = we / 100.0 if we > 1.5 else we
    co = _f(r.get("dev_coachability"), 0.6)
    co = co / 100.0 if co > 1.5 else co
    # Normalised so a bought-in 22-year-old with average habits lands near 1.0.
    return a * (0.7 + 0.6 * we) * (0.8 + 0.4 * co) * buy_in * LEVELS[level]["pace"] / 1.15


def _store(session: Any) -> Dict[str, Any]:
    st = getattr(session, "development_goals", None)
    if not isinstance(st, dict):
        st = {}
        session.development_goals = st
    return st


def _player(session: Any, pid: str) -> Any:
    try:
        from app.sim_engine.franchise.storyline_engine import _player_from_roster  # noqa: WPS433

        return _player_from_roster(session, str(pid))
    except Exception:
        return None


def _last(name: str) -> str:
    return str(name or "He").split()[-1]


def _grow_keys(player: Any, keys: List[str], points: float) -> float:
    """Add `points` (0-99 scale) to each key; returns the average actually applied."""
    r = _ratings(player)
    applied = []
    for k in keys:
        if k not in r:
            continue
        v = _f(r[k])
        unit = 0.01 if v <= 1.5 else 1.0
        hi = 1.0 if unit == 0.01 else 99.0
        new = min(hi, v + points * unit)
        r[k] = new
        applied.append((new - v) / unit)
    inv = getattr(player, "_invalidate_ovr_memo", None)
    if callable(inv):
        inv()
    cache = getattr(player, "_gm_runtime_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    return sum(applied) / len(applied) if applied else 0.0


def _all_keys(player: Any) -> List[str]:
    keys: List[str] = []
    for fid in eligible_focus(player):
        for k in FOCUS[fid]["keys"]:
            if k not in keys:
                keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Meeting templates
# ---------------------------------------------------------------------------


def availability(session: Any, ctx: Dict[str, Any]) -> Dict[str, bool]:
    player = ctx.get("player")
    pid = str(ctx.get("player_id") or "")
    if player is None:
        return {"dev": False, "pot": False}
    row = _store(session).get(pid) or {}
    season = _season(session)
    dev_active = any(g.get("status") == "active" for g in row.get("dev", []))
    dev_count = sum(1 for g in row.get("dev", []) if int(g.get("season") or 0) == season)
    pot = row.get("pot") or {}
    pot_this_season = int(pot.get("season") or -1) == season
    return {
        "dev": not dev_active and dev_count < MAX_DEV_GOALS_PER_SEASON and bool(eligible_focus(player)),
        "pot": _age(player) <= 27 and not pot_this_season,
    }


def build_templates(session: Any, ctx: Dict[str, Any], gm_choice: Any) -> List[Dict[str, Any]]:
    player = ctx.get("player")
    if player is None:
        return []
    avail = availability(session, ctx)
    choices = []
    for fid in suggested_focus(player):
        spec = FOCUS[fid]
        choices.append(gm_choice(
            f"focus_{fid}", f"{spec['label']} (now {area_score(player, fid):.0f})",
            f"Make {spec['label'].lower()} the goal. His {spec['label'].lower()} is one of the weaker parts of his game.",
            {"profile_changes": {"actor": {"state.focus": 2}}, "relationship": {"communication": 2}, "devgoal_focus": fid},
        ))
    pot_choices = []
    for opt in season_goal_options(session, player, str(ctx.get("player_id") or "")):
        spec = METRICS[opt["metric"]]
        now = f" (now {opt['base_txt']})" if opt["base_txt"] else ""
        detail = (f"Judged on his real numbers over {SEASON_WINDOW_DAYS} days. "
                  + ("Based on his season so far." if opt["from_stats"] else "Early season: baseline estimated from his ratings.")
                  + (" A big ask: a bigger ceiling raise if he hits it." if opt["level"] == "ambitious" else " Hitting it raises his ceiling."))
        c = gm_choice(f"pot_{opt['metric']}", f"{spec['label']}: {opt['target_txt']}{now}", detail,
                      {"profile_changes": {"actor": {"state.focus": 2}}, "relationship": {"communication": 2},
                       "potgoal_metric": opt["metric"], "potgoal_target": opt["target"], "potgoal_baseline": opt["baseline"],
                       "potgoal_level": opt["level"]})
        c["say"] = spec["say"].format(target_txt=opt["target_txt"], base_txt=opt["base_txt"] or "nothing yet")
        pot_choices.append(c)
    return [
        {"id": "set_development_goal", "category": "development", "title": "Set a development goal",
         "player_line": "What do you want me working on?", "choices": choices, "available": avail["dev"] and bool(choices)},
        {"id": "set_potential_goal", "category": "development", "title": "Set a season stat goal",
         "player_line": "Where do you see my game going this year?", "choices": pot_choices, "available": avail["pot"] and bool(pot_choices)},
    ]


def gm_say(itype: str, cid: str) -> Optional[str]:
    if itype == "set_development_goal" and cid.startswith("focus_"):
        lines = FOCUS_LINES.get(cid[6:])
        return lines[0][0] if lines else None
    return None


def reply(itype: str, cid: str, warm: bool, seed: str) -> Optional[str]:
    pool: Tuple[str, ...] = ()
    if itype == "set_development_goal" and cid.startswith("focus_"):
        lines = FOCUS_LINES.get(cid[6:])
        if lines:
            pool = lines[1] if warm else lines[2]
    elif itype == "set_potential_goal" and cid.startswith("pot_") and cid[4:] in METRICS:
        pool = METRICS[cid[4:]]["warm" if warm else "cold"]
    if not pool:
        return None
    return _rng(seed, "goal_reply").choice(pool)


def ask_line(itype: str, seed: str) -> str:
    pool = DEV_ASK if itype == "set_development_goal" else POT_ASK
    return _rng(seed, "goal_ask").choice(pool)


def followups(mc: Dict[str, Any], base_choice: Dict[str, Any], reaction: str, scale_outcome: Any, add: Any) -> List[Dict[str, Any]]:
    """Beat 2. Development goal: pick the size of the target. Potential goal: pick the reward."""
    itype = str(mc.get("itype") or "")
    base = dict(base_choice.get("outcome") or {})
    cid = str(base_choice.get("id") or "c")
    out: List[Dict[str, Any]] = []
    if itype == "set_development_goal":
        fid = str(base.get("devgoal_focus") or "")
        label = FOCUS.get(fid, {}).get("label", "it").lower()
        rows = [
            ("steady", f"+2 in four weeks", f"Two points better in your {label} in four weeks. Very doable.",
             f"Two points in four weeks. I'll have that done early.", "He'll get there", "safe", -6),
            ("stretch", f"+4 in five weeks", f"Four points better in five weeks. That's a real jump, and if you hit it your ceiling goes up.",
             f"Four points in five weeks. Okay. That's a real target.", "A real push", "uncertain", -2),
            ("ambitious", f"+6 in six weeks", f"Six points in six weeks. It's a lot, but if you get there you're a different player.",
             f"Six in six weeks. That's a lot. I'm going to chase it.", "He could fall short", "risky", 4),
        ]
        for lid, btn, say, close, read, risk, td in rows:
            oc = {k: v for k, v in base.items() if not str(k).startswith("devgoal")}
            oc = scale_outcome(dict(oc), 1.0, 1.0) if oc else {}
            oc["devgoal"] = {"focus": fid, "level": lid}
            cold = {"steady": "Fine. Two points. I'll get it done.", "stretch": "Four points. Fine. I'll do the work.",
                    "ambitious": "Six. If that's what you want, I'll try."}[lid]
            out.append({"id": f"{cid}__{lid}", "label": btn, "say": say, "close": close, "close_cold": cold,
                        "detail": {"steady": "Small, likely. No ceiling bonus.", "stretch": "Hit it and his potential rises a little.",
                                   "ambitious": "Hit it and his potential rises more. Miss and it costs morale and trust."}[lid],
                        "outcome": oc, "tension_delta": td, "read": read, "risk": risk, "tone": lid})
        return out
    for rid, btn, say, close in POT_REWARDS:
        oc = {k: v for k, v in base.items() if not str(k).startswith("potgoal")}
        oc["potgoal"] = {"metric": base.get("potgoal_metric"), "target": base.get("potgoal_target"),
                         "baseline": base.get("potgoal_baseline"), "level": base.get("potgoal_level") or "stretch", "reward": rid}
        detail = {"role": "Hit it and you owe him a promotion. A promise is logged when he hits it.",
                  "contract": "Hit it and he expects a contract talk. Big morale lift if he gets there.",
                  "growth": "No promise. Less motivating for ambitious players."}[rid]
        td = -6 if rid != "growth" else (4 if _f(mc.get("ambition"), 50) >= 65 else -2)
        cold = {"role": "Fine. Hit it and I move up. Don't forget.", "contract": "Okay. A contract talk. I'll hold you to it.",
                "growth": "Sure. For me, then."}[rid]
        out.append({"id": f"{cid}__{rid}", "label": btn, "say": say, "close": close, "close_cold": cold, "detail": detail, "outcome": oc,
                    "tension_delta": td, "read": "He'll like that" if rid != "growth" else "He may want more",
                    "risk": "safe" if rid != "growth" else "uncertain", "tone": rid})
    return out


# ---------------------------------------------------------------------------
# Applying and ticking
# ---------------------------------------------------------------------------


def apply_from_meeting(session: Any, ctx: Dict[str, Any], outcome: Dict[str, Any], scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    player = ctx.get("player")
    pid = str(ctx.get("player_id") or "")
    name = str((ctx.get("entity") or {}).get("player_name") or getattr(player, "name", "") or "He")
    if player is None or not pid:
        return {}
    if (scene or {}).get("walked_out"):
        return {"ok": False, "message": f"{_last(name)} walked out before you could set the goal."}
    reaction = str((scene or {}).get("reaction") or "receptive")
    buy_in = BUY_IN.get(reaction, 0.9)
    day, season = _day(session), _season(session)
    row = _store(session).setdefault(pid, {"dev": [], "pot": None})
    dg = outcome.get("devgoal")
    if isinstance(dg, dict) and dg.get("focus") in FOCUS and dg.get("level") in LEVELS:
        target, days = LEVELS[dg["level"]]["dev"]
        start = area_score(player, dg["focus"])
        goal = {"focus": dg["focus"], "level": dg["level"], "target": target, "start_score": start, "gained": 0.0,
                "start_day": day, "due_day": day + days, "season": season, "buy_in": round(buy_in, 2),
                "status": "active", "player_name": name}
        row["dev"] = [g for g in row.get("dev", []) if g.get("status") != "active"] + [goal]
        row["dev"] = row["dev"][-8:]
        lab = FOCUS[dg["focus"]]["label"]
        tone = {"opens_up": "He's fully bought in.", "receptive": "He's on board.", "guarded": "He's not fully sold, so progress will be slower.",
                "bristles": "He resents it. Expect slow progress.", "boils_over": "He left angry. Don't expect much."}.get(reaction, "")
        return {"ok": True, "message": f"Goal set: {_last(name)}'s {lab.lower()} {start:.0f} → {start + target:.0f} in {days} days. {tone}".strip(),
                "goal": goal}
    pg = outcome.get("potgoal")
    if isinstance(pg, dict) and pg.get("metric") in METRICS:
        m = pg["metric"]
        goal = {"metric": m, "level": pg.get("level") or "stretch", "reward": pg.get("reward") or "growth",
                "target": float(pg["target"]), "baseline": float(pg["baseline"]), "start_stats": _stat_row(session, pid),
                "start_day": day, "due_day": day + SEASON_WINDOW_DAYS, "extended": False, "season": season,
                "buy_in": round(buy_in, 2), "status": "active", "player_name": name, "start_ovr": round(_ovr99(player), 1)}
        row["pot"] = goal
        return {"ok": True, "goal": goal,
                "message": f"Season goal set: {_last(name)} chases {_fmt(m, goal['target'])} over the next {SEASON_WINDOW_DAYS} days, judged on his real numbers."}
    return {}


def _feed(session: Any, pid: str, text: str, tone: str) -> None:
    ent = (getattr(session, "universe_players", None) or {}).get(str(pid))
    if not isinstance(ent, dict):
        return
    feed = list(ent.get("life_feed") or [])
    iso = ""
    try:
        from app.sim_engine.franchise.storyline_engine import _u_current_meta  # noqa: WPS433

        iso = str(_u_current_meta(session)[1])
    except Exception:
        pass
    feed.append({"iso": iso, "text": text, "tone": tone})
    ent["life_feed"] = feed[-40:]


def _nudge_state(session: Any, pid: str, **deltas: float) -> None:
    ent = (getattr(session, "universe_players", None) or {}).get(str(pid))
    st = ent.get("state") if isinstance(ent, dict) else None
    if not isinstance(st, dict):
        return
    for k, d in deltas.items():
        st[k] = max(0.0, min(100.0, _f(st.get(k), 55) + d))


def _raise_potential(player: Any, points: float) -> float:
    if points <= 0:
        return 0.0
    r = _ratings(player)
    for k in ("dev_potential", "dev_ceiling"):
        if k in r:
            v = _f(r[k])
            r[k] = min(1.0, v + points / 100.0) if v <= 1.5 else min(99.0, v + points)
    pot = _f(getattr(player, "potential", 0), 0)
    if pot:
        try:
            player.potential = min(1.0, pot + points / 100.0) if pot <= 1.5 else min(99.0, pot + points)
        except Exception:
            pass
    return points


def _news(session: Any, pid: str, name: str, headline: str, summary: str) -> None:
    try:
        import uuid

        from app.sim_engine.franchise.storyline_engine import _u_record_storyline  # noqa: WPS433

        event = {"id": f"uve_{uuid.uuid4().hex[:10]}", "kind": "development_goal", "team_id": str(getattr(session, "user_team_id", "") or ""),
                 "participants": [pid], "player_id": pid, "player_name": name, "event_tier": "minor", "tone": "positive"}
        _u_record_storyline(session, event=event, headline=headline, summary=summary, cause_type="DEVELOPMENT_GOAL",
                            category="development", heat=34, public=True)
    except Exception:
        pass


def _promise_role(session: Any, pid: str) -> None:
    try:
        from app.sim_engine.franchise.storyline_engine import create_universe_promise  # noqa: WPS433

        create_universe_promise(session, {"type": "role_opportunity", "player_id": pid, "due_games": 6,
                                          "description": "Promotion after hitting his season goal.",
                                          "success_readiness": 0.4, "failure_readiness": -2.0}, pid, f"goal_{pid}")
    except Exception:
        pass


def tick(session: Any, elapsed_days: int = 1) -> int:
    """Daily growth toward active goals, and resolution at the deadline. Returns goals resolved."""
    store = getattr(session, "development_goals", None)
    if not isinstance(store, dict) or not store:
        return 0
    day = _day(session)
    days = max(1, min(14, int(elapsed_days or 1)))
    resolved = 0
    for pid, row in list(store.items()):
        player = _player(session, pid)
        for g in row.get("dev", []):
            if g.get("status") != "active":
                continue
            if player is None:
                g["status"] = "void"
                continue
            span = max(1, int(g["due_day"]) - int(g["start_day"]))
            rate = float(g["target"]) / span * _capacity(player, float(g.get("buy_in") or 1.0), g["level"])
            ceiling_room = _ceiling99(player) + 6 - area_score(player, g["focus"])
            if ceiling_room < 0:
                rate *= 0.35
            noise = _rng(pid, g["focus"], day).uniform(0.55, 1.45)
            step = rate * days * noise
            if step > 0:
                g["gained"] = round(float(g.get("gained") or 0) + _grow_keys(player, list(FOCUS[g["focus"]]["keys"]), step), 2)
            if day >= int(g["due_day"]):
                resolved += 1
                _resolve_dev(session, pid, player, g)
        g = row.get("pot")
        if isinstance(g, dict) and g.get("status") == "active" and g.get("metric") in METRICS:
            if player is None:
                g["status"] = "void"
                continue
            # The work behind the number: modest real growth in the areas that drive it.
            areas = [f for f in METRICS[g["metric"]]["areas"] if f in FOCUS and f in eligible_focus(player)]
            keys = [k for f in areas for k in FOCUS[f]["keys"]]
            if keys:
                rate = 3.0 / SEASON_WINDOW_DAYS * _capacity(player, float(g.get("buy_in") or 1.0), g["level"])
                step = rate * days * _rng(pid, "pot", day).uniform(0.6, 1.4)
                g["growth"] = round(float(g.get("growth") or 0) + _grow_keys(player, keys, step), 2)
            value, games = season_view_value(session, pid, g)
            g["value"] = value
            g["games"] = games
            if day >= int(g["due_day"]):
                if games < int(METRICS[g["metric"]].get("min_gp", 8)) and not g.get("extended"):
                    g["due_day"] = int(g["due_day"]) + SEASON_EXTENSION_DAYS
                    g["extended"] = True
                    continue
                resolved += 1
                _resolve_pot(session, pid, player, g)
    return resolved


def _resolve_dev(session: Any, pid: str, player: Any, g: Dict[str, Any]) -> None:
    gained, target = float(g.get("gained") or 0), float(g["target"])
    name = g.get("player_name") or "He"
    lab = FOCUS[g["focus"]]["label"].lower()
    frac = gained / target if target else 0
    if frac >= 0.95:
        g["status"] = "hit"
        bonus = _raise_potential(player, LEVELS[g["level"]]["potential_bonus"])
        g["potential_gain"] = bonus
        _nudge_state(session, pid, morale=4, gm_trust=3, confidence=4)
        extra = f" His ceiling went up {bonus:.0f}." if bonus else ""
        _feed(session, pid, f"Hit his {lab} goal: +{gained:.1f} in {int(g['due_day']) - int(g['start_day'])} days.{extra}", "positive")
        if g["level"] != "steady":
            _news(session, pid, name, f"{name}'s extra work on his {lab} is paying off",
                  f"{name} set a {lab} target with the front office and beat it. Coaches say the difference shows in games.")
    elif frac >= 0.6:
        g["status"] = "partial"
        _nudge_state(session, pid, morale=1)
        _feed(session, pid, f"Came close on his {lab} goal: +{gained:.1f} of +{target:.0f}.", "neutral")
    else:
        g["status"] = "missed"
        hit = {"steady": (-1, 0), "stretch": (-3, -1), "ambitious": (-5, -3)}[g["level"]]
        _nudge_state(session, pid, morale=hit[0], gm_trust=hit[1], confidence=hit[0])
        _feed(session, pid, f"Fell short of his {lab} goal: +{gained:.1f} of +{target:.0f}.", "negative")


def _resolve_pot(session: Any, pid: str, player: Any, g: Dict[str, Any]) -> None:
    name = g.get("player_name") or "He"
    m = g["metric"]
    value, games = season_view_value(session, pid, g)
    g["value"], g["games"] = value, games
    if value is None or games < int(METRICS[m].get("min_gp", 8)):
        g["status"] = "void"
        _feed(session, pid, f"His {METRICS[m]['label'].lower()} goal ran out without enough games to judge it.", "neutral")
        return
    prog = _season_progress(g, value)
    hit = (value - float(g["target"])) * METRICS[m]["dir"] >= 0
    if hit:
        g["status"] = "hit"
        bonus = _raise_potential(player, 1.0 + LEVELS[g["level"]]["potential_bonus"])
        g["potential_gain"] = bonus
        _nudge_state(session, pid, morale=5, gm_trust=4, confidence=5)
        if g.get("reward") == "role":
            _promise_role(session, pid)
        elif g.get("reward") == "contract":
            _nudge_state(session, pid, belonging=4)
        _feed(session, pid, f"Hit his season goal: {_fmt(m, value)} (target {_fmt(m, float(g['target']))}). His ceiling rose {bonus:.0f}.", "positive")
        headline = {"points": f"{name} hits the scoring pace the club set for him",
                    "goals": f"{name} finds the net like the front office asked",
                    "xgf_pct": f"{name}'s numbers show he's tilting the ice",
                    "xga60": f"{name} becomes the shutdown piece the club wanted",
                    "hits_blocks": f"{name} has become a nightmare to play against",
                    "faceoffs": f"{name} turns himself into a faceoff weapon",
                    "save_pct": f"{name} gives the club the goaltending it asked for",
                    "gsax": f"{name} has been stealing games",
                    "quality_starts": f"{name} has been a rock in net"}.get(m, f"{name} reaches the target the club set for him")
        _news(session, pid, name, headline, f"{name} set a target with the front office: {_fmt(m, float(g['target']))}. "
              f"Over {games} games he got to {_fmt(m, value)}.")
    elif prog >= 0.5:
        g["status"] = "partial"
        bonus = _raise_potential(player, 0.5 if g["level"] == "ambitious" else 0.0)
        g["potential_gain"] = bonus
        _nudge_state(session, pid, morale=1)
        _feed(session, pid, f"Came close on his season goal: {_fmt(m, value)} of {_fmt(m, float(g['target']))}.", "neutral")
    else:
        g["status"] = "missed"
        mood = {"stretch": (-3, -2), "ambitious": (-5, -3)}.get(g["level"], (-1, 0))
        if g.get("reward") in ("role", "contract"):
            mood = (mood[0] - 1, mood[1])
        _nudge_state(session, pid, morale=mood[0], gm_trust=mood[1], confidence=mood[0])
        _feed(session, pid, f"Missed his season goal: {_fmt(m, value)}, target was {_fmt(m, float(g['target']))}.", "negative")


def goals_view(session: Any, player_id: str) -> List[Dict[str, Any]]:
    """Rows for the player panel (same shape as directives_view), active and recent."""
    row = (getattr(session, "development_goals", None) or {}).get(str(player_id)) or {}
    day = _day(session)
    out: List[Dict[str, Any]] = []
    for g in list(row.get("dev", []))[-3:]:
        lab = FOCUS.get(g.get("focus"), {}).get("label", "Skill")
        frac = min(1.0, float(g.get("gained") or 0) / max(0.1, float(g["target"])))
        st = g.get("status")
        eff = (f"+{float(g.get('gained') or 0):.1f} of +{float(g['target']):.0f} · {max(0, int(g['due_day']) - day)} days left"
               if st == "active" else {"hit": "Hit it", "partial": "Came close", "missed": "Missed", "void": "Ended"}.get(st, st))
        if st == "hit" and g.get("potential_gain"):
            eff += f" · ceiling +{float(g['potential_gain']):.0f}"
        out.append({"slot": f"devgoal_{g.get('focus')}_{g.get('start_day')}", "kind": "goal", "label": f"{lab} goal ({LEVELS[g['level']]['label'].lower()})",
                    "strength": frac, "accepted": st, "effect": eff, "progress": frac, "status": st})
    g = row.get("pot")
    if isinstance(g, dict) and g.get("metric") in METRICS:
        m = g["metric"]
        value, games = season_view_value(session, str(player_id), g)
        frac = min(1.0, _season_progress(g, value))
        st = g.get("status")
        reward = {"role": " · reward: bigger role", "contract": " · reward: contract talk"}.get(g.get("reward"), "")
        if st == "active":
            eff = (f"Now {_fmt(m, value)} over {games} games · target {_fmt(m, float(g['target']))} · "
                   f"{max(0, int(g['due_day']) - day)} days left{reward}") if value is not None else \
                  f"Target {_fmt(m, float(g['target']))} · no games yet · {max(0, int(g['due_day']) - day)} days left{reward}"
        else:
            eff = ({"hit": "Hit it", "partial": "Came close", "missed": "Missed", "void": "Not enough games"}.get(st, st)
                   + (f": {_fmt(m, value)} vs {_fmt(m, float(g['target']))}" if value is not None else ""))
            if g.get("potential_gain"):
                eff += f" · ceiling +{float(g['potential_gain']):.1f}"
        out.append({"slot": "potgoal", "kind": "goal", "label": f"Season goal: {METRICS[m]['label']}",
                    "strength": frac, "accepted": st, "effect": eff, "progress": frac, "status": st})
    return out


# ===========================================================================
# Season goals v2: a real stat target that fits his game, judged on his actual
# numbers over the goal window. Scorers chase points; players who aren't known
# for scoring chase the analytics and role numbers that measure their value.
# ===========================================================================

SEASON_WINDOW_DAYS = 60
SEASON_EXTENSION_DAYS = 21

# metric -> spec. "dir" +1 higher is better, -1 lower is better.
METRICS: Dict[str, Dict[str, Any]] = {
    "points": {"label": "Points pace", "dir": 1, "areas": ("shooting", "playmaking", "hockey_iq"), "min_gp": 8,
               "say": "I want you producing at {target_txt}. You're at {base_txt} right now.",
               "warm": ("I want that. I think I've got more points in me.", "That's the number I've been chasing anyway."),
               "cold": ("That's a lot of points for the minutes I'm getting.", "I'll try. I need the power play for that.")},
    "goals": {"label": "Goal pace", "dir": 1, "areas": ("shooting", "puck_skills"), "min_gp": 8,
              "say": "I want you scoring at {target_txt}. You're on {base_txt} now. Shoot more, get to the net.",
              "warm": ("A goal pace. Yeah. I want to be the guy who scores.", "I like that. I've been passing up too many."),
              "cold": ("Goals come in bunches. That's hard to promise.", "Okay. I'll shoot more.")},
    "shots": {"label": "Shot volume", "dir": 1, "areas": ("shooting", "skating"), "min_gp": 8,
              "say": "Forget the goals for now. I want {target_txt} from you. Volume, every night.",
              "warm": ("Shots I can control. I like that.", "Okay. Get it to the net. I can do that."),
              "cold": ("Shots without goals doesn't mean much.", "Okay. More shots.")},
    "xgf_pct": {"label": "Expected-goals share", "dir": 1, "areas": ("hockey_iq", "defence", "playmaking"), "min_gp": 8,
                "say": "Points aside, when you're on the ice I want us winning the chances: {target_txt}. You're at {base_txt}.",
                "warm": ("That's how I want to be judged. I drive play even when it doesn't show up on the scoresheet.", "I like that. Somebody finally looks past the points."),
                "cold": ("Analytics. Okay. The guys I play with matter for that too.", "I'll try. That's not all on me.")},
    "xga60": {"label": "Shutdown: expected goals against", "dir": -1, "areas": ("defence", "hockey_iq"), "min_gp": 8,
              "say": "Your job is keeping their chances down. I want our expected goals against at {target_txt} when you're out there. You're at {base_txt}.",
              "warm": ("A shutdown number. Yeah. That's my game.", "Good. That's the job I want."),
              "cold": ("So I'm a checker now.", "Okay. Keep it out of our net.")},
    "hits_blocks": {"label": "Hits and blocks", "dir": 1, "areas": ("physical", "defence"), "min_gp": 8,
                    "say": "I want you hard to play against: {target_txt}. You're at {base_txt}.",
                    "warm": ("Hits and blocks. I'll pay the price.", "That's my game. Easy."),
                    "cold": ("I'm not just a body.", "Okay. I'll throw more hits.")},
    "faceoffs": {"label": "Faceoff percentage", "dir": 1, "areas": ("faceoffs",), "min_gp": 8, "min_draws": 60,
                 "say": "I want you winning {target_txt} of your draws. You're at {base_txt}. Start us with the puck.",
                 "warm": ("Draws. Yeah. I can get there.", "Okay. That's a quick way to earn more ice."),
                 "cold": ("Faceoffs are half luck.", "Okay. Draws.")},
    "pk": {"label": "Penalty-kill role", "dir": 1, "areas": ("defence", "consistency"), "min_gp": 8,
           "say": "Earn the coaches' trust on the kill. I want you at {target_txt}. You're at {base_txt}.",
           "warm": ("The kill. I'd love to own that.", "Okay. I'll block everything they shoot."),
           "cold": ("PK minutes aren't up to me.", "Okay.")},
    "save_pct": {"label": "Save percentage", "dir": 1, "areas": ("g_positioning", "g_reflexes"), "min_gp": 5,
                 "say": "I want you at {target_txt}. You're at {base_txt}. Be the reason we win.",
                 "warm": ("That's a starter's number. I want it.", "Okay. I'll get there."),
                 "cold": ("Save percentage depends on the guys in front of me.", "Okay.")},
    "gsax": {"label": "Goals saved above expected", "dir": 1, "areas": ("g_positioning", "g_rebounds", "g_mental"), "min_gp": 5,
             "say": "I want you {target_txt} over the next two months. Steal us a few.",
             "warm": ("Steal a few. Yeah. That's the job.", "Okay. I'll stop the ones I shouldn't."),
             "cold": ("That's a lot of pressure.", "Okay.")},
    "quality_starts": {"label": "Quality starts", "dir": 1, "areas": ("g_mental", "g_positioning"), "min_gp": 5,
                       "say": "Give us a chance every night: {target_txt}. You're at {base_txt}.",
                       "warm": ("Consistency. That's what I want too.", "Okay. Every night."),
                       "cold": ("One bad night and it's gone.", "Okay.")},
}

_COUNTERS = ("gp", "g", "a", "pts", "sog", "hit", "blk", "toi_sec", "pk_toi_sec", "fow", "fol", "xgf", "xga",
             "analytics_gp", "saves", "shots_against", "ga", "goalie_xga", "quality_starts", "starts", "gs")


def _stat_row(session: Any, pid: str) -> Dict[str, float]:
    row = ((getattr(session, "player_season_stats", None) or {}).get(str(pid)) or {})
    return {k: _f(row.get(k), 0.0) for k in _COUNTERS}


def _rate(metric: str, s: Dict[str, float]) -> Optional[float]:
    gp = s.get("gp", 0.0)
    starts = s.get("starts") or s.get("gs") or gp
    if metric == "points":
        return s["pts"] / gp if gp else None
    if metric == "goals":
        return s["g"] / gp * 82 if gp else None
    if metric == "shots":
        return s["sog"] / gp if gp else None
    if metric == "xgf_pct":
        t = s["xgf"] + s["xga"]
        return s["xgf"] / t if t > 0 else None
    if metric == "xga60":
        hrs = s["toi_sec"] / 3600.0
        return 5.0 * s["xga"] / hrs if hrs > 0.5 else None
    if metric == "hits_blocks":
        return (s["hit"] + s["blk"]) / gp if gp else None
    if metric == "faceoffs":
        t = s["fow"] + s["fol"]
        return s["fow"] / t if t > 0 else None
    if metric == "pk":
        return s["pk_toi_sec"] / 60.0 / gp if gp else None
    if metric == "save_pct":
        return s["saves"] / s["shots_against"] if s["shots_against"] > 0 else None
    if metric == "gsax":
        return s["goalie_xga"] - s["ga"] if gp else None
    if metric == "quality_starts":
        return s["quality_starts"] / starts if starts else None
    return None


def _fmt(metric: str, v: Optional[float]) -> str:
    if v is None:
        return "no number yet"
    return {
        "points": f"{v:.2f} points a game",
        "goals": f"a {v:.0f}-goal pace",
        "shots": f"{v:.1f} shots a game",
        "xgf_pct": f"{v * 100:.1f}% of expected goals",
        "xga60": f"{v:.2f} expected goals against per 60",
        "hits_blocks": f"{v:.1f} hits and blocks a game",
        "faceoffs": f"{v * 100:.1f}% on faceoffs",
        "pk": f"{v:.1f} PK minutes a game",
        "save_pct": f"a {v:.3f} save percentage".replace("0.", "."),
        "gsax": f"{v:+.1f} goals saved above expected",
        "quality_starts": f"quality starts in {v * 100:.0f}% of games",
    }[metric]


def _window_delta(start: Dict[str, float], now: Dict[str, float]) -> Dict[str, float]:
    return {k: max(0.0, now.get(k, 0.0) - start.get(k, 0.0)) for k in _COUNTERS}


def _group(player: Any, fids: Tuple[str, ...]) -> float:
    vals = [area_score(player, f) for f in fids if f in FOCUS and all(k in _ratings(player) for k in FOCUS[f]["keys"])]
    return sum(vals) / len(vals) if vals else 0.0


def _default_base(metric: str, player: Any) -> float:
    pos = _pos(player)
    d = pos.startswith("D")
    ovr = _ovr99(player)
    try:
        from app.sim_engine.progression.development import _expected_ppg_for_ovr  # noqa: WPS433

        ppg = float(_expected_ppg_for_ovr(ovr))
    except Exception:
        ppg = max(0.1, (ovr - 60) / 40)
    if d:
        ppg *= 0.6
    phys = _group(player, ("physical",)) or 70.0
    r = _ratings(player)
    return {
        "points": ppg,
        "goals": ppg * (0.3 if d else 0.42) * 82,
        "shots": (1.6 if d else 2.1) + (_group(player, ("shooting",)) - 70) * 0.04,
        "xgf_pct": 0.50,
        "xga60": 2.6,
        "hits_blocks": (2.6 if d else 1.4) + (phys - 70) * 0.05,
        "faceoffs": 0.45 + ((lambda v: v / 100.0 if v > 1.5 else v)(_f(r.get("def_faceoffs"), 0.7)) - 0.7) * 0.4,
        "pk": 1.0,
        "save_pct": 0.905 + (_group(player, ("g_positioning", "g_reflexes")) - 80) * 0.0015,
        "gsax": 0.0,
        "quality_starts": 0.55,
    }[metric]


def _target(metric: str, base: float, big: bool) -> float:
    if metric in ("points", "goals", "shots", "hits_blocks"):
        return base * (1.25 if big else 1.15) + (0.05 if metric == "points" else 0)
    if metric == "xgf_pct":
        return base + (0.025 if big else 0.015)
    if metric == "xga60":
        return base * (0.90 if big else 0.94)
    if metric == "faceoffs":
        return base + (0.03 if big else 0.02)
    if metric == "pk":
        return base + (0.6 if big else 0.4)
    if metric == "save_pct":
        return base + (0.006 if big else 0.004)
    if metric == "gsax":
        return base + (4.0 if big else 2.0)
    if metric == "quality_starts":
        return base + (0.08 if big else 0.05)
    return base


def _is_scorer(player: Any, season_ppg: Optional[float]) -> bool:
    d = _pos(player).startswith("D")
    if season_ppg is not None and season_ppg >= (0.42 if d else 0.55):
        return True
    off = _group(player, ("shooting", "playmaking", "puck_skills"))
    dfn = _group(player, ("defence", "physical"))
    return off >= dfn + 3 and _ovr99(player) >= 76


def season_goal_options(session: Any, player: Any, pid: str) -> List[Dict[str, Any]]:
    pos = _pos(player)
    stats = _stat_row(session, pid)
    has_stats = stats["gp"] >= 10
    if pos.startswith("G"):
        metrics = ["save_pct", "gsax", "quality_starts"]
    else:
        season_ppg = stats["pts"] / stats["gp"] if has_stats else None
        d = pos.startswith("D")
        if _is_scorer(player, season_ppg):
            metrics = ["points", "xgf_pct", "shots"] if d else ["points", "goals", "xgf_pct"]
        else:
            metrics = ["xga60", "xgf_pct", "hits_blocks", "pk"] if d else ["xgf_pct", "xga60", "hits_blocks", "shots"]
        if pos == "C":
            metrics = metrics[:3] + ["faceoffs"]
    big = _age(player) <= 23 or (_ceiling99(player) - _ovr99(player)) >= 5
    out = []
    for m in metrics[:4]:
        base = _rate(m, stats) if has_stats else None
        if base is None:
            base = _default_base(m, player)
        if m == "gsax":
            base = 0.0
        tgt = _target(m, base, big)
        out.append({"metric": m, "baseline": round(base, 4), "target": round(tgt, 4), "level": "ambitious" if big else "stretch",
                    "base_txt": _fmt(m, base) if m != "gsax" else "", "target_txt": _fmt(m, tgt), "from_stats": has_stats})
    return out


def season_view_value(session: Any, pid: str, g: Dict[str, Any]) -> Tuple[Optional[float], int]:
    delta = _window_delta(g.get("start_stats") or {}, _stat_row(session, pid))
    return _rate(g["metric"], delta), int(delta.get("starts") or delta.get("gs") or delta.get("gp") or 0)


def _season_progress(g: Dict[str, Any], value: Optional[float]) -> float:
    if value is None:
        return 0.0
    base, tgt, d = float(g["baseline"]), float(g["target"]), METRICS[g["metric"]]["dir"]
    span = (tgt - base) * d
    if span <= 0:
        return 1.0 if (value - tgt) * d >= 0 else 0.0
    return max(0.0, min(1.25, (value - base) * d / span))
