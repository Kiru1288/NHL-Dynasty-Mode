"""Adaptive headline/body templates for data-driven franchise storylines."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

HEADLINES: Dict[str, List[str]] = {
    "star_underperforming": [
        "{name} enters witness protection program on the scoresheet",
        "Local ${cap:.1f}M forward reportedly still searching for the net",
        "{name}'s scoring lamp remains cold",
        "Top-line {role} producing at the pace of a decorative traffic cone",
        "The ${cap:.1f}M man has gone missing",
        "{name}'s drought becoming impossible to ignore",
        "Scoring drought becoming impossible to ignore",
    ],
    "rookie_breakout": [
        "{name} refuses to wait politely for his development timeline",
        "Rookie {name} is forcing the coaching staff to rewrite the depth chart",
        "{team} rookie {name} outpacing every internal projection",
    ],
    "superstar_carrying": [
        "{name} accidentally becomes the entire offense",
        "{name} is the only reason {team}'s record isn't worse",
        "Star {name} carrying {team} like it's a second job",
    ],
    "contract_pressure": [
        "Cap hit remains elite, production has filed a missing persons report",
        "${cap:.1f}M AAV vs {pts} points — the math isn't mathing",
        "{name}'s contract is louder than his box score",
    ],
    "goalie_meltdown": [
        "Goalie has apparently decided the puck deserves freedom",
        "{name} letting in goals like the net is a suggestion",
        "{team} crease in crisis — {name} at {sv:.3f} SV%",
    ],
    "goalie_heater": [
        "Netminder commits grand larceny, steals another two points",
        "{name} rolling at elite save percentage over {gp} starts",
        "{team} goalie {name} is stealing games nightly",
    ],
    "backup_taking_net": [
        "Backup goalie has politely stolen the starter's office keys",
        "{name} forcing a crease conversation in {team}",
        "Starter seat getting warm — {name} pushing for more starts",
    ],
    "surprise_team": [
        "{team} keeps winning despite the league's best attempts to explain it away",
        "Surprise contender alert: {team} sitting {record}",
        "{team} outperforming every preseason projection",
    ],
    "contender_collapse": [
        "{team} begins annual tradition of making everyone nervous",
        "Paper contender {team} stuck at {record}",
        "{team} collapse has the room playing tight",
    ],
    "playoff_race": [
        "Every point now feels like it comes with a stress tax",
        "{team} bubble watch — {record} and no margin for error",
        "Wild-card math getting cruel for {team}",
    ],
    "losing_skid": [
        "{team} skid has the room playing like the clock is guilty",
        "{team} losing streak turning the dressing room tense",
        "Slide continues — {team} at {record}",
    ],
    "win_streak": [
        "{team} rolling like they found a cheat code for effort",
        "{team} on a heater at {record}",
        "Win streak energy building around {team}",
    ],
}

BODIES: Dict[str, List[str]] = {
    "star_underperforming": [
        "{name} ({ovr:.0f} OVR) has {pts} points in {gp} games ({ppg:.2f} P/GP) vs ~{exp_pts:.0f} expected for role.",
        "Production pace {ppg:.2f} P/GP is well below expected {exp_ppg:.2f} for a {ovr:.0f}-overall {role}.",
    ],
    "rookie_breakout": [
        "{name} is outpacing development expectations with {pts} points in {gp} games.",
        "Rookie scoring pace {ppg:.2f} P/GP exceeds expected {exp_ppg:.2f}.",
    ],
    "superstar_carrying": [
        "{name} is producing like a star on a team sitting {record}.",
        "Elite individual scoring on a sub-.500 club.",
    ],
    "contract_pressure": [
        "${cap:.2f}M AAV vs {pts} points in {gp} games.",
        "High cap hit with well-below-expected production.",
    ],
    "goalie_meltdown": [
        "{name} at {sv:.3f} SV% and {gaa:.2f} GAA over {gp} starts.",
        "Save percentage well below expected baseline for {ovr:.0f} OVR.",
    ],
    "goalie_heater": [
        "{name} rolling at {sv:.3f} SV% over {gp} starts.",
        "Save percentage materially above expected baseline.",
    ],
    "backup_taking_net": [
        "{name} is pushing for a larger share of starts in {team}.",
        "Backup performance is forcing a starter conversation.",
    ],
    "surprise_team": [
        "{team} is outperforming expectations at {record}.",
        "Strong run reflected in {record} record.",
    ],
    "contender_collapse": [
        "Strong on-paper club is {record} through {gp} games.",
        "High expectations meet a losing record.",
    ],
    "playoff_race": [
        "{team} sits around the wild-card line at {record}.",
        "Late-season standings place the club near the cut line.",
    ],
    "losing_skid": [
        "Team record {record} with mounting losses.",
        "Multiple losses in current standings snapshot.",
    ],
    "win_streak": [
        "Strong run reflected in {record} record.",
        "Standings show sustained winning.",
    ],
}


def story_ctx(**kwargs: Any) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        ctx[key] = value
    if "name" not in ctx and "player_name" in ctx:
        ctx["name"] = ctx["player_name"]
    if "team" not in ctx and "team_name" in ctx:
        ctx["team"] = ctx["team_name"]
    return ctx


def _format_line(template: str, ctx: Dict[str, Any]) -> str:
    safe = defaultdict(str)
    for key, value in (ctx or {}).items():
        safe[key] = value
    try:
        return str(template).format_map(safe).strip()
    except (KeyError, ValueError, TypeError):
        return str(template).strip()


def pick_line(rng: random.Random, stype: str, ctx: Dict[str, Any], body: bool = False) -> str:
    pool: List[str] = list(BODIES.get(str(stype), []) if body else HEADLINES.get(str(stype), []))
    if not pool:
        name = str(ctx.get("name") or ctx.get("player_name") or "Player")
        team = str(ctx.get("team") or ctx.get("team_name") or "Team")
        if body:
            return f"{name} storyline developing around {team}."
        return f"{name} — developing story for {team}"
    template = rng.choice(pool)
    return _format_line(template, ctx)


def classify_story_lane(
    *,
    cause_type: str = "",
    category: str = "",
    stype: str = "",
    severity: str = "",
    priority: str = "",
    heat: int = 0,
    knowledge_type: str = "",
    public_level: str = "",
    legal_status: str = "",
    incident_family: str = "",
    **_: Any,
) -> str:
    kt = str(knowledge_type or "").lower()
    pl = str(public_level or "").lower()
    ct = str(cause_type or "").upper()
    st = str(stype or "").lower()

    if kt in {"claim", "rumor", "rumour"} or pl in {"rumour", "rumor", "unconfirmed", "whisper"}:
        return "rumor"
    if kt in {"fact", "confirmed"} or pl in {"reported", "confirmed", "verified"}:
        return "recap"
    if "TRADE" in ct or "RUMOR" in ct or "trade" in st:
        return "rumor"
    if "CONDUCT" in ct or legal_status or incident_family:
        return "recap"
    if int(heat or 0) >= 68 and str(priority or "").upper() in {"HIGH", "CRITICAL"}:
        return "recap"
    if str(severity or "").lower() == "major":
        return "recap"
    return "default"


def lane_flags(lane: str) -> Dict[str, Any]:
    lane_key = str(lane or "default")
    return {
        "story_lane": lane_key,
        "is_rumor": lane_key == "rumor",
        "is_recap": lane_key == "recap",
        "is_fact_lane": lane_key == "recap",
    }
