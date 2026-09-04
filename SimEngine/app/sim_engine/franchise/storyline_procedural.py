"""Procedural narrative copy and life-event rule engine.

Headlines and summaries are composed from player life state, personality tags,
and numeric gates — not static pick_line pools or catalog dicts.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.sim_engine.franchise.storyline_copy import (
    format_sv_pct,
    normalize_save_pct,
    valid_goalie_heater_sv,
    valid_goalie_meltdown_sv,
)

ScoreFn = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]], float]
PredicateFn = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]], bool]

_SCORE_LABELS = {
    "character": "Character",
    "volatility": "Volatility",
    "personal_stress": "Personal stress",
    "role_satisfaction": "Role satisfaction",
    "media_stress": "Media stress",
    "tension": "Tension",
    "media_score": "Media pressure score",
    "production_ppg": "Points per game",
    "expected_ppg": "Expected P/GP",
    "overall": "Overall",
    "points_delta": "Points vs expected",
    "save_pct": "Save %",
    "expected_save_pct": "Expected save %",
    "team_points_pct": "Team points %",
    "streak": "Streak length",
    "family_orientation": "Family orientation",
    "home_stability": "Home stability",
    "sleep_quality": "Sleep quality",
    "community_connection": "Community connection",
    "contract_satisfaction": "Contract satisfaction",
    "winning_satisfaction": "Winning satisfaction",
    "room_value": "Locker-room value",
    "candidate_score": "Trigger score",
}


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return lo if value < lo else hi if value > hi else value


def _name(entity: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> str:
    if ctx and ctx.get("name"):
        return str(ctx["name"])
    return str(entity.get("player_name") or "Player")


def _state(entity: Dict[str, Any]) -> Dict[str, Any]:
    return dict(entity.get("state") or {})


def _personality(entity: Dict[str, Any]) -> Dict[str, Any]:
    return dict(entity.get("personality") or {})


def _life(entity: Dict[str, Any]) -> Dict[str, Any]:
    return dict(entity.get("life") or {})


def _tags(entity: Dict[str, Any]) -> List[str]:
    tags = list(entity.get("personality_tags") or [])
    for row in entity.get("niche_abilities") or []:
        t = str(row.get("id") or "")
        if t and t not in tags:
            tags.append(t)
    for n in entity.get("niche_ids") or entity.get("niches") or []:
        t = str(n)
        if t and t not in tags:
            tags.append(t)
    return tags[:6]


def build_trigger_context(
    *,
    cause_type: str = "",
    rule_id: str = "",
    stype: str = "",
    kind: str = "",
    scores: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    tags: Optional[List[str]] = None,
    reporter: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "cause_type": cause_type,
        "rule_id": rule_id,
        "stype": stype,
        "kind": kind,
        "scores": {k: round(float(v), 2) for k, v in (scores or {}).items()},
        "thresholds": {k: round(float(v), 2) for k, v in (thresholds or {}).items()},
        "tags": list(tags or []),
        "reporter": reporter,
    }
    ctx.update(extra)
    ctx["reason_lines"] = trigger_reason_lines(ctx)
    ctx["reason_text"] = format_trigger_reason(ctx)
    return ctx


def trigger_reason_lines(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    cause = str(ctx.get("cause_type") or ctx.get("stype") or ctx.get("kind") or "").strip()
    if cause:
        lines.append({"code": "cause", "label": "Cause", "value": cause.replace("_", " ")})
    scores = dict(ctx.get("scores") or {})
    thresholds = dict(ctx.get("thresholds") or {})
    for key, val in scores.items():
        label = _SCORE_LABELS.get(key, key.replace("_", " ").title())
        thr = thresholds.get(key)
        if thr is not None:
            lines.append({"code": key, "label": label, "value": f"{val:.0f} (gate {thr:.0f})"})
        else:
            lines.append({"code": key, "label": label, "value": f"{val:.0f}"})
    for tag in ctx.get("tags") or []:
        lines.append({"code": "tag", "label": str(tag).replace("_", " "), "value": None})
    reporter = str(ctx.get("reporter") or "").strip()
    if reporter:
        lines.append({"code": "reporter", "label": "Reporter voice", "value": reporter})
    rule_id = str(ctx.get("rule_id") or "").strip()
    if rule_id:
        lines.append({"code": "rule", "label": "Life rule", "value": rule_id.replace("_", " ")})
    return lines[:8]


def format_trigger_reason(ctx: Dict[str, Any]) -> str:
    parts: List[str] = []
    for line in trigger_reason_lines(ctx):
        val = line.get("value")
        if val:
            parts.append(f"{line['label']}: {val}")
        elif line.get("label"):
            parts.append(str(line["label"]))
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Procedural performance-story copy (replaces pick_line pools)
# ---------------------------------------------------------------------------

def compose_data_story_copy(stype: str, ctx: Dict[str, Any], rng: random.Random, *, body: bool = False) -> str:
    name = str(ctx.get("name") or "Player")
    team = str(ctx.get("team") or "the club")
    role = str(ctx.get("role") or "player")
    ovr = float(ctx.get("ovr") or 0)
    gp = int(ctx.get("gp") or 0)
    pts = int(ctx.get("pts") or 0)
    ppg = float(ctx.get("ppg") or 0)
    exp_pts = float(ctx.get("exp_pts") or 0)
    exp_ppg = exp_pts / max(1, gp)
    record = str(ctx.get("record") or "")
    cap = float(ctx.get("cap") or 0)
    age = int(ctx.get("age") or 25)
    goals = int(ctx.get("goals") or 0)
    save_pct = normalize_save_pct(ctx.get("save_pct"), ctx.get("sv"), ctx.get("sv_pct"))
    exp_sv = normalize_save_pct(ctx.get("expected_save_pct"), ctx.get("exp_sv"))
    sv_txt = format_sv_pct(save_pct)
    exp_txt = format_sv_pct(exp_sv)
    streak = int(ctx.get("streak") or 0)
    gaa = float(ctx.get("gaa") or 0)
    rank = int(ctx.get("league_rank") or ctx.get("rank") or 0)
    opponent = str(ctx.get("opponent") or "")
    contract_year = bool(ctx.get("contract_year"))

    if stype == "star_underperforming":
        if body:
            return (
                f"{name} ({round(ovr)} OVR) has {pts} points in {gp} games ({ppg:.2f} P/GP). "
                f"For a {role} at that rating the model expected roughly {exp_ppg:.2f} P/GP ({exp_pts:.0f} total)."
            )
        lead = rng.choice(
            [
                f"{name}'s scoring pace ({ppg:.2f} P/GP) trails his {round(ovr)}-OVR profile",
                f"Production gap: {name} at {pts} PTS through {gp} GP vs {exp_pts:.0f} projected",
                f"{team} star {name} running well below expected output ({ppg:.2f} vs {exp_ppg:.2f} P/GP)",
            ]
        )
        return lead

    if stype == "rookie_breakout":
        if body:
            return (
                f"{name} ({age}) is producing at {ppg:.2f} P/GP across {gp} games — "
                f"above the {exp_ppg:.2f} P/GP baseline for his profile."
            )
        return rng.choice(
            [
                f"Rookie surge: {name} at {ppg:.2f} P/GP after {gp} games",
                f"{name} outpacing development curve ({pts} PTS, {ppg:.2f} P/GP)",
                f"Young {role} {name} forcing a larger role conversation",
            ]
        )

    if stype == "superstar_carrying":
        rank_bit = f", {rank}th in the league" if rank else ""
        if body:
            extra = f" on a {record} club{rank_bit}" if record else ""
            if contract_year:
                extra += " with a contract year looming"
            return (
                f"{name} is driving {team}'s offense with {pts} points in {gp} games"
                f"{extra}. Teammates have not matched that pace."
            )
        options = [
            f"{name} carrying {team}'s attack ({pts} PTS in {gp} GP)",
            f"Offensive load falling on {name} ({ppg:.2f} P/GP)",
            f"{team} leaning heavily on {name}'s production",
        ]
        if record:
            options.append(f"{name} producing through a {record} slog for {team}")
        if rank >= 20:
            options.append(f"Award-race case: {name} keeping {team} relevant")
        if contract_year:
            options.append(f"Contract-year heater: {name} carrying {team}")
        if ctx.get("pp_pts"):
            options.append(f"PP engine: {name} generating {team}'s extra-man offense")
        return rng.choice(options)

    if stype == "contract_pressure":
        if body:
            return (
                f"{name} is in a contract year with a ${cap/1_000_000:.2f}M cap hit and "
                f"{'strong' if ppg >= exp_ppg else 'uneven'} counting stats ({ppg:.2f} P/GP)."
            )
        return rng.choice(
            [
                f"Contract-year pressure building on {name}",
                f"{name}'s next deal in focus as cap hit sits at ${cap/1_000_000:.2f}M",
                f"Negotiation cloud forming around {name}",
            ]
        )

    if stype == "goalie_meltdown":
        if not valid_goalie_meltdown_sv(save_pct):
            return ""
        if body:
            baseline = f" vs a {exp_txt} expected baseline" if exp_txt else ""
            gaa_bit = f" and {gaa:.2f} GAA" if gaa else ""
            return f"{name}'s save percentage ({sv_txt}){gaa_bit} is well below profile{baseline} across {gp} starts."
        return rng.choice(
            [
                f"Goaltending concern: {name} at {sv_txt} through {gp} GP",
                f"{name}'s form ({sv_txt} SV%) lagging expected {exp_txt or 'baseline'}",
                f"{team} net unsettled as {name} struggles ({sv_txt})",
            ]
        )

    if stype == "goalie_heater":
        if not valid_goalie_heater_sv(save_pct):
            return ""
        if body:
            gaa_bit = f" and {gaa:.2f} GAA" if gaa else ""
            extra = f" against a {exp_txt} expected mark" if exp_txt else ""
            return f"{name} is outperforming his profile at {sv_txt}{gaa_bit} over {gp} games{extra}."
        options = [
            f"Hot goaltending: {name} at {sv_txt}",
            f"{name} stealing games for {team} ({sv_txt} SV%)",
            f"Net confidence rising behind {name} ({sv_txt})",
        ]
        if gaa:
            options.append(f"{name} at {sv_txt} / {gaa:.2f} GAA through {gp} starts")
        if record:
            options.append(f"{team}'s crease story: {name} {sv_txt} while club sits {record}")
        return rng.choice(options)

    if stype == "backup_taking_net":
        if body:
            return f"Usage trends suggest {name}'s workload is shifting as the backup pushes for more starts."
        return rng.choice(
            [
                f"Net competition intensifies around {name}",
                f"{team} goaltending rotation in flux",
                f"Backup pushing {name} for starts",
            ]
        )

    if stype in ("surprise_team", "hot_streak_team"):
        if body:
            return f"{team} sits at {record} with momentum building across the roster."
        return rng.choice(
            [
                f"{team} exceeding expectations ({record})",
                f"Surprise surge: {team} at {record}",
                f"{team}'s fast start drawing league attention",
            ]
        )

    if stype in ("contender_collapse", "cold_streak_team"):
        if body:
            return f"{team} has dropped to {record} after a difficult stretch."
        return rng.choice(
            [
                f"{team} sliding ({record})",
                f"Expectations clash with results for {team}",
                f"{team}'s form raising internal questions",
            ]
        )

    if stype == "playoff_race":
        if body:
            return f"{team} ({record}) is locked in a tight playoff positioning fight."
        return f"Playoff-race pressure on {team} ({record})"

    if stype == "losing_skid":
        if body:
            return f"{team} has dropped {streak} straight ({record})."
        return f"{team} on a {streak}-game skid ({record})"

    if stype == "win_streak":
        if body:
            return f"{team} has won {streak} straight ({record})."
        return f"{team} riding a {streak}-game win streak"

    if stype == "goal_drought":
        if body:
            return f"{name} has {goals} goals in {gp} games — below the expected rate for a {round(ovr)}-OVR {role}."
        return f"Scoring drought: {name} with {goals} G in {gp} GP"

    if stype == "veteran_fade":
        if body:
            return f"Veteran {name} ({age}) is producing at {ppg:.2f} P/GP, below his historical baseline."
        return f"Age curve questions around {name} ({ppg:.2f} P/GP)"

    if body:
        return f"{name} storyline developing around {team}."
    return f"{name} — developing story for {team}"


_COMMUNITY_HOOKS = (
    "a hospital visit with the {team} Foundation",
    "a local rink clinic for kids",
    "the club's community skate",
    "a charity food-drive stop downtown",
    "a veterans' hospital appearance",
    "an after-school hockey program",
)


def community_event_copy(name: str, team: str, player_id: str = "") -> Tuple[str, str]:
    seed = abs(hash(str(player_id or name)))
    hook = _COMMUNITY_HOOKS[seed % len(_COMMUNITY_HOOKS)].format(team=team or "the club")
    headline = f"{name} spends a day on {hook}"
    summary = f"{name} spent time on {hook}. Teammates say those days still matter in the room."
    return headline, summary


def compose_shutout_copy(
    *,
    goalie_name: str,
    team: str,
    opponent: str = "",
    record: str = "",
    league_rank: int = 0,
    prior_shutouts: int = 0,
    snapped_skid: bool = False,
) -> Tuple[str, str]:
    gname = str(goalie_name or "").strip() or "the starter"
    team_n = str(team or "the club")
    opp = str(opponent or "the opponent")
    if prior_shutouts >= 2:
        headline = f"{gname} posts shutout No. {prior_shutouts + 1} in a week — Vezina buzz for {team_n}"
        summary = f"Coaches are pointing to {gname} as the reason {team_n} is stacking zeros. {opp} did not solve him."
        return headline, summary
    if prior_shutouts >= 1:
        headline = f"{gname} blanks {opp} again as {team_n} ride the crease"
        summary = f"Back-to-back shutout form from {gname}. The room is rallying around the starter."
        return headline, summary
    if snapped_skid:
        headline = f"{gname} snaps {team_n}'s skid with a shutout of {opp}"
        summary = f"{team_n} needed a reset. {gname} provided it with a clean sheet."
        return headline, summary
    if 1 <= league_rank <= 3:
        headline = f"First-place {team_n}: {gname} shuts out {opp}"
        summary = f"At {record or 'the top of the table'}, {gname} added another shutout to a contender's crease."
        return headline, summary
    if 7 <= league_rank <= 16:
        headline = f"Wildcard-race shutout: {gname} blanks {opp}"
        summary = f"{team_n} ({record or 'in the mix'}) grabbed a huge point-race result behind {gname}."
        return headline, summary
    headline = f"{gname} shuts out {opp}"
    summary = f"{team_n} blanked {opp} behind {gname}."
    return headline, summary


_REPORTER_FRAMES = (
    (
        "coverage",
        "{actor} confronts {reporter} over a hit piece",
        "{actor} challenges {reporter} ({outlet}) after a week of critical coverage.",
        "You're turning every answer into a crisis.",
    ),
    (
        "trade_rumor",
        "{actor} corners {reporter} about trade chatter",
        "{actor} is angry {reporter} ({outlet}) put his name in a trade rumor without sourcing the room.",
        "You don't get to shop me in print.",
    ),
    (
        "coach_criticism",
        "{actor} pushes back on {reporter}'s coaching take",
        "{actor} tells {reporter} ({outlet}) to stop using him as a proxy to bury the staff.",
        "Leave the bench out of it. Talk to me.",
    ),
    (
        "contract",
        "{actor} clashes with {reporter} over contract noise",
        "{actor} is tired of {reporter} ({outlet}) framing every shift as leverage for the next deal.",
        "I'm playing hockey. Stop writing my negotiation.",
    ),
    (
        "leak",
        "{actor} accuses {reporter} of locker-room leaks",
        "{actor} believes {reporter} ({outlet}) is printing closed-door details that never should have left the room.",
        "Someone in here is talking. Don't print it like it's nothing.",
    ),
    (
        "treatment",
        "{actor} calls out {reporter} for poor media treatment",
        "{actor} says {reporter} ({outlet}) has been baiting him in scrums and twisting the quotes.",
        "Ask the question once. Don't ambush me.",
    ),
)


def reporter_conflict_copy(
    rng: random.Random,
    actor_name: str,
    reporter_name: str,
    outlet: str,
    *,
    physical: bool = False,
    player_id: str = "",
) -> Dict[str, str]:
    if physical:
        return {
            "frame": "altercation",
            "title": f"Media hallway altercation involving {actor_name}",
            "summary": (
                f"A heated exchange between {actor_name} and {reporter_name} ({outlet}) "
                f"turns into a brief shoving incident before security intervenes."
            ),
            "player_line": "It had been building. I'm not pretending it hadn't.",
        }
    seed = abs(hash(f"{player_id}|{actor_name}|{reporter_name}"))
    if rng is not None:
        seed ^= rng.randrange(1, 10_000)
    idx = seed % len(_REPORTER_FRAMES)
    _fid, title_t, summary_t, line = _REPORTER_FRAMES[idx]
    ctx = {"actor": actor_name, "reporter": reporter_name, "outlet": outlet or "media"}
    return {
        "frame": _fid,
        "title": title_t.format(**ctx),
        "summary": summary_t.format(**ctx),
        "player_line": line,
    }


def reporter_followup_copy(
    actor_name: str,
    reporter_name: str,
    outlet: str,
    frame: str = "",
) -> Tuple[str, str]:
    outlet_n = outlet or "the outlet"
    if frame == "trade_rumor":
        return (
            f"{reporter_name} answers {actor_name}: the rumor stays in print",
            f"{reporter_name} ({outlet_n}) declined to walk back the trade chatter. Teammates are telling {actor_name} to let it die.",
        )
    if frame == "leak":
        return (
            f"Room split after {actor_name}'s leak accusation",
            f"Veterans privately warned {actor_name} that going after {reporter_name} over a leak makes the dressing room smaller.",
        )
    return (
        f"{reporter_name} responds to {actor_name}'s scrum",
        f"{reporter_name} ({outlet_n}) said the questions were fair. {actor_name}'s teammates are treating it as a one-day story.",
    )


def build_data_story_trigger_context(
    stype: str,
    ctx: Dict[str, Any],
    evidence: Dict[str, Any],
    cause: str = "",
) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    thresholds: Dict[str, float] = {}
    gp = max(1, int(ctx.get("gp") or evidence.get("games_played") or 1))
    ppg = float(ctx.get("ppg") or evidence.get("points_per_game") or 0)
    exp_pts = float(ctx.get("exp_pts") or evidence.get("expected_points") or 0)
    exp_ppg = exp_pts / gp
    ovr = float(ctx.get("ovr") or evidence.get("overall") or 0)

    if stype == "star_underperforming":
        scores["production_ppg"] = ppg
        scores["expected_ppg"] = exp_ppg
        scores["overall"] = ovr
        scores["points_delta"] = float(evidence.get("points") or ctx.get("pts") or 0) - exp_pts
        thresholds["production_ppg"] = exp_ppg * 0.62
    elif stype == "rookie_breakout":
        scores["production_ppg"] = ppg
        scores["expected_ppg"] = exp_ppg
        scores["overall"] = ovr
        thresholds["production_ppg"] = max(0.55, exp_ppg * 1.45)
    elif stype in ("goalie_meltdown", "goalie_heater"):
        sv_n = normalize_save_pct(evidence.get("save_pct"), ctx.get("save_pct"), ctx.get("sv"))
        exp_n = normalize_save_pct(evidence.get("expected_save_pct"), ctx.get("expected_save_pct"), ctx.get("exp_sv"))
        if sv_n is not None:
            scores["save_pct"] = sv_n
        if exp_n is not None:
            scores["expected_save_pct"] = exp_n
    elif stype in ("losing_skid", "win_streak"):
        scores["streak"] = float(ctx.get("streak") or evidence.get("streak") or 0)
        thresholds["streak"] = 3.0
    elif stype in ("surprise_team", "contender_collapse", "playoff_race"):
        scores["team_points_pct"] = float(evidence.get("points_pct") or ctx.get("points_pct") or 0)

    cause_type = {
        "star_underperforming": "PLAYER_LOW_PRODUCTION",
        "rookie_breakout": "ROOKIE_BREAKOUT",
        "superstar_carrying": "SUPERSTAR_CARRY",
        "contract_pressure": "CONTRACT_DISPUTE",
        "goalie_meltdown": "GOALIE_BAD_FORM",
        "goalie_heater": "GOALIE_HEATER",
        "losing_skid": "LOSING_STREAK",
        "win_streak": "WINNING_STREAK",
    }.get(stype, stype.upper())

    return build_trigger_context(
        cause_type=cause_type,
        stype=stype,
        scores=scores,
        thresholds=thresholds,
        cause_detail=cause[:200] if cause else "",
    )


def pick_line_with_trigger(
    rng: random.Random,
    stype: str,
    ctx: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    *,
    body: bool = False,
    cause: str = "",
) -> Tuple[str, Dict[str, Any]]:
    text = compose_data_story_copy(stype, ctx, rng, body=body)
    trigger = build_data_story_trigger_context(stype, ctx, evidence or {}, cause)
    return text, trigger


# ---------------------------------------------------------------------------
# Life-event rule engine (replaces MINOR_*_LIFE_EVENTS catalogs)
# ---------------------------------------------------------------------------

def _partnered(life: Dict[str, Any]) -> bool:
    return str(life.get("relationship_status") or "single") not in ("single", "")


def _compose_life_headline(rule_id: str, entity: Dict[str, Any], fields: Dict[str, float], positive: bool) -> str:
    name = _name(entity)
    stress = fields.get("personal_stress", 0)
    if rule_id == "family_stress_spillover":
        return f"Home-life pressure is following {name} to the rink (stress {stress:.0f})"
    if rule_id == "sleep_recovery_hit":
        return f"Poor recovery is showing up in {name}'s daily routine"
    if rule_id == "media_backlash":
        return f"Online noise is getting under {name}'s skin"
    if rule_id == "relocation_family_strain":
        return f"Family adjustment strain is weighing on {name}"
    if rule_id == "housing_instability":
        return f"Unsettled home life is adding background stress for {name}"
    if rule_id == "partner_friction":
        return f"Relationship tension is spilling into {name}'s week"
    if rule_id == "financial_pinch":
        return f"Off-ice expenses are nagging at {name}"
    if rule_id == "community_lift":
        headline, _ = community_event_copy(name, str((entity.get("team_name") or entity.get("team") or "the club")), str(entity.get("player_id") or entity.get("id") or ""))
        return headline
    if rule_id == "family_milestone":
        return f"Positive family news is energizing {name}"
    if rule_id == "belonging_reset":
        return f"{name} is feeling more rooted in the city"
    if rule_id == "mentorship_click":
        return f"A veteran conversation is clicking for {name}"
    if rule_id == "confidence_surge":
        return f"Confidence is trending up for {name}"
    if positive:
        return f"A personal bright spot for {name}"
    return f"Off-ice friction affecting {name}'s focus"


def _compose_life_summary(rule_id: str, entity: Dict[str, Any], fields: Dict[str, float], positive: bool) -> str:
    tags = _tags(entity)
    tag_note = f" Profile tags: {', '.join(tags[:3])}." if tags else ""
    if rule_id == "family_stress_spillover":
        return (
            f"Elevated personal stress ({fields.get('personal_stress', 0):.0f}) with strong family orientation "
            f"({fields.get('family_orientation', 0):.0f}) is competing with hockey focus.{tag_note}"
        )
    if rule_id == "media_backlash":
        return (
            f"Media stress ({fields.get('media_stress', 0):.0f}) and volatility ({fields.get('volatility', 0):.0f}) "
            f"are amplifying off-ice distraction.{tag_note}"
        )
    if rule_id == "community_lift":
        _, summary = community_event_copy(
            _name(entity),
            str((entity.get("team_name") or entity.get("team") or "the club")),
            str(entity.get("player_id") or entity.get("id") or ""),
        )
        return summary + tag_note
    if positive:
        return f"Life-state signals aligned for a small positive off-ice moment.{tag_note}"
    return f"Life-state pressure crossed the threshold for a minor negative off-ice event.{tag_note}"


def _life_rule(
    rule_id: str,
    *,
    positive: bool,
    predicate: PredicateFn,
    weight: ScoreFn,
    profile: Dict[str, float],
    ovr: float,
    days: int,
    heat: int,
    mutation_id: str = "",
    requires_partnered: bool = False,
    requires_dependents: bool = False,
    requires_single: bool = False,
    attrs: Optional[Dict[str, float]] = None,
    character: float = 0.0,
    public_chance: float = 0.0,
    potential_chance: float = 0.0,
    potential: float = 0.0,
    leave: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": mutation_id or rule_id,
        "rule_id": rule_id,
        "positive": positive,
        "predicate": predicate,
        "weight": weight,
        "profile": profile,
        "ovr": ovr,
        "attrs": dict(attrs or {}),
        "days": days,
        "heat": heat,
        "character": character,
        "requires_partnered": requires_partnered,
        "requires_dependents": requires_dependents,
        "requires_single": requires_single,
        "public_chance": public_chance,
        "potential_chance": potential_chance,
        "potential": potential,
        "leave": leave,
    }


def _life_fields(entity: Dict[str, Any]) -> Dict[str, float]:
    state = _state(entity)
    life = _life(entity)
    p = _personality(entity)
    concerns = dict(entity.get("concerns") or {})
    contract = dict(concerns.get("contract") or {})
    winning = dict(concerns.get("winning") or {})
    return {
        "personal_stress": float(state.get("personal_stress", 25)),
        "media_stress": float(state.get("media_stress", 25)),
        "role_satisfaction": float(state.get("role_satisfaction", 60)),
        "character": float(p.get("character", 55)),
        "volatility": float(p.get("volatility", 40)),
        "family_orientation": float(p.get("family_orientation", 50)),
        "media_savvy": float(p.get("media_savvy", 50)),
        "money_focus": float(p.get("money_focus", 45)),
        "home_stability": float(life.get("home_stability", 65)),
        "sleep_quality": float(life.get("sleep_quality", 68)),
        "relocation_strain": float(life.get("relocation_strain", 15)),
        "community_connection": float(life.get("community_connection", 35)),
        "financial_stress": float(life.get("financial_stress", 10)),
        "contract_satisfaction": float(contract.get("satisfaction", 60)),
        "winning_satisfaction": float(winning.get("satisfaction", 55)),
        "belonging": float(state.get("belonging", 55)),
        "confidence": float(state.get("confidence", 55)),
        "energy": float(state.get("energy", 70)),
    }


def _build_life_event_rules() -> List[Dict[str, Any]]:
    def f(entity, life, state, p):
        return _life_fields(entity)

    return [
        _life_rule(
            "family_stress_spillover",
            positive=False,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["personal_stress"] > 68 and f(e, life, state, p)["family_orientation"] > 55
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["personal_stress"] + f(e, life, state, p)["family_orientation"] * 0.35,
            profile={"state.personal_stress": 4.0, "state.focus": -2.5, "state.morale": -2.0},
            ovr=-0.55,
            days=6,
            heat=11,
            mutation_id="family_difficulty",
            requires_dependents=True,
        ),
        _life_rule(
            "sleep_recovery_hit",
            positive=False,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["sleep_quality"] < 52 and f(e, life, state, p)["energy"] < 58
            ),
            weight=lambda e, life, state, p: (62 - f(e, life, state, p)["sleep_quality"]) + (58 - f(e, life, state, p)["energy"]),
            profile={"state.energy": -3.0, "state.focus": -2.0},
            ovr=-0.4,
            days=2,
            heat=7,
            mutation_id="poor_sleep",
            attrs={"stamina": -0.35},
        ),
        _life_rule(
            "media_backlash",
            positive=False,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["media_stress"] > 58
                and f(e, life, state, p)["volatility"] > 52
                and f(e, life, state, p)["character"] < 62
            ),
            weight=lambda e, life, state, p: (
                f(e, life, state, p)["media_stress"] + f(e, life, state, p)["volatility"] * 0.4 + (62 - f(e, life, state, p)["character"]) * 0.5
            ),
            profile={"state.media_stress": 3.0, "state.confidence": -1.5},
            ovr=-0.35,
            days=5,
            heat=13,
            mutation_id="social_media_noise",
            character=-0.15,
            attrs={"discipline": -0.2},
        ),
        _life_rule(
            "relocation_family_strain",
            positive=False,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["relocation_strain"] > 42 and int(life.get("dependents") or 0) > 0
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["relocation_strain"] + 20,
            profile={"state.personal_stress": 4.0, "state.belonging": -2.0},
            ovr=-0.5,
            days=9,
            heat=10,
            mutation_id="child_relocation",
            requires_dependents=True,
        ),
        _life_rule(
            "housing_instability",
            positive=False,
            predicate=lambda e, life, state, p: f(e, life, state, p)["home_stability"] < 48,
            weight=lambda e, life, state, p: 70 - f(e, life, state, p)["home_stability"],
            profile={"state.personal_stress": 2.0, "state.energy": -1.0},
            ovr=-0.25,
            days=4,
            heat=6,
            mutation_id="home_repairs",
        ),
        _life_rule(
            "partner_friction",
            positive=False,
            predicate=lambda e, life, state, p: (
                _partnered(life) and f(e, life, state, p)["personal_stress"] > 60 and f(e, life, state, p)["home_stability"] < 55
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["personal_stress"] + (55 - f(e, life, state, p)["home_stability"]),
            profile={"state.morale": -2.0, "state.personal_stress": 3.0},
            ovr=-0.35,
            days=4,
            heat=9,
            mutation_id="partner_argument",
            requires_partnered=True,
        ),
        _life_rule(
            "financial_pinch",
            positive=False,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["financial_stress"] > 38 or (
                    f(e, life, state, p)["money_focus"] > 62 and f(e, life, state, p)["contract_satisfaction"] < 45
                )
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["financial_stress"] + max(0, f(e, life, state, p)["money_focus"] - 55),
            profile={"state.personal_stress": 2.0, "state.morale": -1.0},
            ovr=-0.2,
            days=5,
            heat=5,
            mutation_id="unexpected_expense",
        ),
        _life_rule(
            "homesick_signal",
            positive=False,
            predicate=lambda e, life, state, p: (
                not _partnered(life) and f(e, life, state, p)["relocation_strain"] > 36 and f(e, life, state, p)["belonging"] < 50
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["relocation_strain"] + (50 - f(e, life, state, p)["belonging"]),
            profile={"state.morale": -2.5, "state.belonging": -2.0},
            ovr=-0.4,
            days=7,
            heat=10,
            mutation_id="homesick",
            requires_single=True,
        ),
        _life_rule(
            "community_lift",
            positive=True,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["community_connection"] > 58 and f(e, life, state, p)["character"] > 55
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["community_connection"] + f(e, life, state, p)["character"] * 0.3,
            profile={"state.morale": 3.0, "state.belonging": 4.0, "state.confidence": 1.0},
            ovr=0.25,
            days=7,
            heat=22,
            mutation_id="charity_success",
            character=0.5,
            public_chance=0.28,
        ),
        _life_rule(
            "family_milestone",
            positive=True,
            predicate=lambda e, life, state, p: (
                _partnered(life) and f(e, life, state, p)["personal_stress"] < 42 and f(e, life, state, p)["family_orientation"] > 58
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["family_orientation"] + (42 - f(e, life, state, p)["personal_stress"]),
            profile={"state.morale": 4.0, "state.belonging": 2.0, "state.personal_stress": -2.0},
            ovr=0.35,
            days=8,
            heat=18,
            mutation_id="pregnancy_news",
            requires_partnered=True,
            public_chance=0.45,
            potential_chance=0.12,
            potential=0.15,
        ),
        _life_rule(
            "belonging_reset",
            positive=True,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["home_stability"] > 68 and f(e, life, state, p)["relocation_strain"] < 28
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["home_stability"] - f(e, life, state, p)["relocation_strain"],
            profile={"state.belonging": 4.0, "state.personal_stress": -3.0, "state.focus": 1.5},
            ovr=0.3,
            days=10,
            heat=12,
            mutation_id="family_settles",
            character=0.2,
            potential_chance=0.15,
            potential=0.2,
        ),
        _life_rule(
            "confidence_surge",
            positive=True,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["confidence"] > 68 and f(e, life, state, p)["winning_satisfaction"] > 60
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["confidence"] + f(e, life, state, p)["winning_satisfaction"] * 0.25,
            profile={"state.confidence": 2.0, "state.morale": 2.0},
            ovr=0.2,
            days=4,
            heat=10,
            mutation_id="fan_moment",
            public_chance=0.35,
        ),
        _life_rule(
            "roots_in_city",
            positive=True,
            predicate=lambda e, life, state, p: (
                f(e, life, state, p)["belonging"] > 62 and not bool(life.get("home_owned"))
            ),
            weight=lambda e, life, state, p: f(e, life, state, p)["belonging"],
            profile={"state.belonging": 4.0, "state.personal_stress": -2.0},
            ovr=0.25,
            days=10,
            heat=16,
            mutation_id="buys_home",
            character=0.25,
            public_chance=0.35,
        ),
        _life_rule(
            "mentorship_click",
            positive=True,
            predicate=lambda e, life, state, p: (
                int(e.get("age", 30) or 30) <= 24 and f(e, life, state, p)["confidence"] < 52 and "mentor" in _tags(e)
            ),
            weight=lambda e, life, state, p: (52 - f(e, life, state, p)["confidence"]) + 20,
            profile={"state.confidence": 2.0, "state.focus": 2.0},
            ovr=0.25,
            days=6,
            heat=10,
            mutation_id="mentor_connection",
            character=0.2,
            attrs={"offensive_awareness": 0.25},
            potential_chance=0.2,
            potential=0.3,
        ),
    ]


_LIFE_EVENT_RULES: List[Dict[str, Any]] = _build_life_event_rules()


def life_event_allowed(entity: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    life = _life(entity)
    relationship = str(life.get("relationship_status") or "single")
    dependents = int(life.get("dependents", 0) or 0)
    if spec.get("requires_partnered") and relationship == "single":
        return False
    if spec.get("requires_dependents") and dependents <= 0:
        return False
    if spec.get("requires_single") and relationship != "single":
        return False
    return True


def evaluate_life_event_rules(
    entity: Dict[str, Any],
    rng: random.Random,
    *,
    positive: bool,
) -> Optional[Dict[str, Any]]:
    """Return the highest-weight matching life-event spec with procedural copy attached."""
    life = _life(entity)
    state = _state(entity)
    p = _personality(entity)
    fields = _life_fields(entity)
    matches: List[Tuple[float, Dict[str, Any]]] = []
    for rule in _LIFE_EVENT_RULES:
        if bool(rule.get("positive")) != bool(positive):
            continue
        if not life_event_allowed(entity, rule):
            continue
        try:
            if not rule["predicate"](entity, life, state, p):
                continue
        except (TypeError, ValueError, KeyError):
            continue
        try:
            weight = float(rule["weight"](entity, life, state, p))
        except (TypeError, ValueError, KeyError):
            weight = 1.0
        matches.append((weight, rule))
    if not matches:
        return None
    matches.sort(key=lambda row: -row[0])
    top_weight = matches[0][0]
    band = [row for row in matches if row[0] >= top_weight - 8]
    rule = rng.choice(band)[1]
    rule_id = str(rule.get("rule_id") or rule.get("id"))
    headline = _compose_life_headline(rule_id, entity, fields, positive)
    summary = _compose_life_summary(rule_id, entity, fields, positive)
    trigger = build_trigger_context(
        cause_type="POSITIVE_LIFE_EVENT" if positive else "MINOR_LIFE_EVENT",
        rule_id=rule_id,
        scores={k: fields[k] for k in fields if k in rule.get("profile", {}) or k in ("personal_stress", "character", "volatility", "family_orientation", "media_stress", "home_stability")},
        thresholds={},
        tags=_tags(entity),
    )
    spec = {k: v for k, v in rule.items() if k not in ("predicate", "weight", "rule_id")}
    spec["headline"] = headline
    spec["summary"] = summary
    spec["trigger_context"] = trigger
    return spec


def build_interaction_trigger_context(
    *,
    kind: str,
    actor: Dict[str, Any],
    target: Optional[Dict[str, Any]] = None,
    score: float = 0.0,
    reporter_name: str = "",
    room_tension: float = 0.0,
    rel_tension: float = 0.0,
) -> Dict[str, Any]:
    state = _state(actor)
    p = _personality(actor)
    scores: Dict[str, float] = {
        "character": float(p.get("character", 55)),
        "volatility": float(p.get("volatility", 40)),
        "personal_stress": float(state.get("personal_stress", 25)),
        "role_satisfaction": float(state.get("role_satisfaction", 60)),
        "media_stress": float(state.get("media_stress", 25)),
        "candidate_score": float(score),
    }
    thresholds: Dict[str, float] = {}
    if kind in ("reporter_confrontation", "reporter_altercation"):
        thresholds["media_stress"] = 58.0
        thresholds["volatility"] = 52.0
        thresholds["character"] = 62.0
    if kind in ("blame_game", "teammate_fight"):
        scores["tension"] = float(rel_tension or room_tension)
        thresholds["tension"] = 68.0
        thresholds["character"] = 50.0
    if kind == "role_frustration" or kind.startswith("request_"):
        thresholds["role_satisfaction"] = 45.0
    if kind == "personal_check_in":
        thresholds["personal_stress"] = 65.0
    cause_type = {
        "reporter_altercation": "PLAYER_REPORTER_ALTERCATION",
        "reporter_confrontation": "PLAYER_REPORTER_CONFRONTATION",
        "teammate_fight": "TEAMMATE_FIGHT",
        "blame_game": "TEAMMATE_CONFLICT",
        "unheralded_leader": "HIGH_CHARACTER_IMPACT",
    }.get(kind, "PLAYER_INTERACTION")
    return build_trigger_context(
        cause_type=cause_type,
        kind=kind,
        scores=scores,
        thresholds=thresholds,
        tags=_tags(actor),
        reporter=reporter_name,
    )
