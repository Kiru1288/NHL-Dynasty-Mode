"""
Player Journeys — career narrative arcs for individual players.

Tracks career phase, peak performance, breakout seasons, decline, awards, and tags.
Also maintains active narrative events (duration, decay) and per-player modifier fields
set via apply_narrative_mechanics_to_rosters() for progression/regression integration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Narrative tags for player journeys
TAG_ROOKIE_SENSATION = "rookie_sensation"
TAG_LATE_BLOOMER = "late_bloomer"
TAG_GENERATIONAL_TALENT = "generational_talent"
TAG_DECLINING_STAR = "declining_star"
TAG_CAREER_RESURGENCE = "career_resurgence"
TAG_PLAYOFF_HERO = "playoff_hero"
TAG_BUST = "bust"
TAG_SUPERSTAR = "superstar"
TAG_BREAKOUT = "breakout"
TAG_BREAKOUT_REPEAT = "breakout_repeat"
TAG_REPEATED_DECLINE = "repeated_decline"
TAG_FALL_FROM_GRACE = "fall_from_grace"
TAG_UNDERDOG_RISE = "underdog_rise"
TAG_VOLATILE_STAR = "volatile_star"

# Narrative states (player career posture for logs / UI)
STATE_BREAKOUT_STAR = "breakout_star"
STATE_RISING_STAR = "rising_star"
STATE_STRUGGLING = "struggling"
STATE_DECLINING_VETERAN = "declining_veteran"
STATE_UNDERDOG = "underdog"
STATE_SUPERSTAR = "superstar"
STATE_STABLE_PRO = "stable_pro"

# --- Mechanical narrative layer (integration with progression / performance proxies) ---
NARRATIVE_MAX_ACTIVE_EVENTS: int = 3
NARRATIVE_MAJOR_COOLDOWN_YEARS: int = 2

_EVT_CONFIDENCE_COLLAPSE = "confidence_collapse"
_EVT_MOMENTUM_HOT = "momentum_hot"
_EVT_MEDIA_STRAIN = "media_spotlight_strain"
_EVT_RESURGENCE_DRIVE = "resurgence_drive"
_EVT_BURNOUT_WEAR = "burnout_wear"
_EVT_LEADERSHIP_SURGE = "leadership_surge"
_EVT_BREAKOUT_PRESSURE = "breakout_expectation_pressure"


def _narrative_event_effect_deltas(kind: str, intensity: float) -> Dict[str, float]:
    """Relative multipliers / shifts; intensity 0..1 scales magnitude."""
    z = max(0.0, min(1.0, float(intensity)))
    z_m = 0.65 + 0.35 * z
    if kind == _EVT_CONFIDENCE_COLLAPSE:
        return {
            "prog_growth": 1.0 - 0.14 * z_m,
            "regression_rate": 1.0 + 0.16 * z_m,
            "breakout_p": max(0.72, 1.0 - 0.20 * z_m),
            "decline_p": 1.0 + 0.14 * z_m,
            "consistency": -0.12 * z_m,
            "variance": 0.10 * z_m,
        }
    if kind == _EVT_MOMENTUM_HOT:
        return {
            "prog_growth": 1.0 + 0.09 * z_m,
            "regression_rate": max(0.82, 1.0 - 0.10 * z_m),
            "breakout_p": 1.0 + 0.10 * z_m,
            "decline_p": max(0.78, 1.0 - 0.12 * z_m),
            "consistency": 0.08 * z_m,
            "variance": 0.05 * z_m,
        }
    if kind == _EVT_MEDIA_STRAIN:
        return {
            "prog_growth": 1.0 - 0.08 * z_m,
            "regression_rate": 1.0 + 0.10 * z_m,
            "breakout_p": max(0.80, 1.0 - 0.08 * z_m),
            "decline_p": 1.0 + 0.08 * z_m,
            "consistency": -0.08 * z_m,
            "variance": 0.12 * z_m,
        }
    if kind == _EVT_RESURGENCE_DRIVE:
        return {
            "prog_growth": 1.0 + 0.11 * z_m,
            "regression_rate": max(0.85, 1.0 - 0.08 * z_m),
            "breakout_p": 1.0 + 0.06 * z_m,
            "decline_p": max(0.85, 1.0 - 0.10 * z_m),
            "consistency": 0.06 * z_m,
            "variance": 0.04 * z_m,
        }
    if kind == _EVT_BURNOUT_WEAR:
        return {
            "prog_growth": 1.0 - 0.12 * z_m,
            "regression_rate": 1.0 + 0.14 * z_m,
            "breakout_p": max(0.75, 1.0 - 0.15 * z_m),
            "decline_p": 1.0 + 0.12 * z_m,
            "consistency": -0.10 * z_m,
            "variance": 0.08 * z_m,
        }
    if kind == _EVT_LEADERSHIP_SURGE:
        return {
            "prog_growth": 1.0 + 0.06 * z_m,
            "regression_rate": max(0.88, 1.0 - 0.06 * z_m),
            "breakout_p": 1.02 + 0.04 * z_m,
            "decline_p": max(0.88, 1.0 - 0.06 * z_m),
            "consistency": 0.05 * z_m,
            "variance": 0.03 * z_m,
        }
    if kind == _EVT_BREAKOUT_PRESSURE:
        return {
            "prog_growth": 1.0 + 0.04 * z_m,
            "regression_rate": 1.0 + 0.04 * z_m,
            "breakout_p": 1.0 + 0.14 * z_m,
            "decline_p": 1.0 + 0.05 * z_m,
            "consistency": -0.05 * z_m,
            "variance": 0.09 * z_m,
        }
    return {
        "prog_growth": 1.0,
        "regression_rate": 1.0,
        "breakout_p": 1.0,
        "decline_p": 1.0,
        "consistency": 0.0,
        "variance": 0.0,
    }


@dataclass
class JourneyProfile:
    """Per-player career narrative profile (evolves season by season)."""
    player_key: str
    name: str
    birth_year: int
    career_phase: str = "unknown"
    peak_ovr: float = 0.0
    breakout_seasons: int = 0
    decline_indicator: bool = False
    seasons_played: int = 0
    prev_ovr: float = 0.0
    prev_prev_ovr: float = 0.0
    awards_won: List[str] = field(default_factory=list)
    championships_won: int = 0
    narrative_tags: List[str] = field(default_factory=list)
    last_rookie_year: Optional[int] = None
    was_declining: bool = False
    # --- Expanded narrative layer (multi-season memory)
    narrative_character: str = "medium"  # low / medium / high (leadership / poise)
    media_pressure_band: str = "medium"  # low / medium / high
    consistency_style: str = "stable"  # streaky / stable / volatile
    clutch_factor: str = "medium"  # low / medium / high
    narrative_state: str = STATE_STABLE_PRO
    ovr_history: List[Tuple[int, float]] = field(default_factory=list)
    major_events: List[Dict[str, Any]] = field(default_factory=list)
    last_team_id: str = ""
    former_team_ids: List[str] = field(default_factory=list)
    decline_seasons: int = 0
    # Active timed narrative modifiers (mechanical); decayed once per year in update_player_journeys
    narrative_active_events: List[Dict[str, Any]] = field(default_factory=list)
    narrative_fired_major_years: Dict[str, int] = field(default_factory=dict)
    # Short buffer for logs / debugging (activate + resolve)
    recent_narrative_signal_logs: List[Dict[str, Any]] = field(default_factory=list)


def _player_key(player: Any) -> str:
    ident = getattr(player, "identity", None)
    name = getattr(ident, "name", None) or getattr(player, "name", "Unknown")
    birth = int(getattr(ident, "birth_year", 0) or 2000)
    return f"{name}|{birth}"


def _player_name(player: Any) -> str:
    ident = getattr(player, "identity", None)
    return getattr(ident, "name", None) or getattr(player, "name", "Unknown")


def _team_name(team: Any) -> str:
    if team is None:
        return "Unknown"
    city = getattr(team, "city", "") or ""
    name = getattr(team, "name", "") or getattr(team, "team_id", "")
    if city and name:
        return f"{city} {name}".strip()
    return str(name or getattr(team, "team_id", "Unknown"))


def _team_id(team: Any) -> str:
    return str(getattr(team, "team_id", None) or getattr(team, "id", "") or "")


def _team_context_phrase(team: Any) -> str:
    if team is None:
        return "his club"
    win = str(getattr(team, "window", getattr(team, "gm_window", "")) or "").lower()
    if win == "rebuild":
        return "a rebuilding room that needs patience and identity"
    if win == "contender":
        return "a team with real championship expectations"
    if win == "emerging":
        return "a young group trying to prove it belongs with the elite"
    if win == "declining":
        return "a veteran core searching for one more push"
    st = str(getattr(team, "status", "") or "").lower()
    if "rebuild" in st:
        return "a franchise still stacking futures"
    if "contend" in st or "powerhouse" in st:
        return "a contender's dressing room"
    return "a team fighting for its place in the standings"


def _append_major_event(profile: JourneyProfile, year: int, kind: str, detail: str) -> None:
    profile.major_events.append({"year": year, "kind": kind, "detail": detail})
    if len(profile.major_events) > 24:
        profile.major_events[:] = profile.major_events[-24:]


def _derive_psych_bands(player: Any) -> Tuple[str, str, str, str]:
    """character, media_pressure, consistency, clutch → each low|medium|high or streaky|stable|volatile."""
    psych = getattr(player, "psych", None)
    if psych is None:
        return ("medium", "medium", "stable", "medium")
    lead = float(getattr(psych, "leadership_emergence", 0.5) or 0.5)
    res = float(getattr(psych, "conflict_resolution", 0.5) or 0.5)
    mot = float(getattr(psych, "internal_motivation", 0.5) or 0.5)
    core = (lead + res + mot) / 3.0
    if core >= 0.62:
        char = "high"
    elif core <= 0.38:
        char = "low"
    else:
        char = "medium"

    media = float(getattr(psych, "media_stress", 0.5) or 0.5) + float(getattr(psych, "fan_pressure", 0.5) or 0.5)
    media *= 0.5
    if media >= 0.62:
        mp = "high"
    elif media <= 0.38:
        mp = "low"
    else:
        mp = "medium"

    vol = float(getattr(psych, "confidence_volatility", 0.5) or 0.5)
    streak = float(getattr(psych, "streak_amplification", 0.5) or 0.5)
    if vol >= 0.58 or streak >= 0.62:
        cstyle = "volatile" if vol >= 0.55 else "streaky"
    elif vol <= 0.38 and streak <= 0.42:
        cstyle = "stable"
    else:
        cstyle = "streaky" if streak > vol else "stable"

    ot = float(getattr(psych, "overtime_composure", 0.5) or 0.5)
    gi = float(getattr(psych, "game_importance_sensitivity", 0.5) or 0.5)
    pn = float(getattr(psych, "playoff_nerves", 0.5) or 0.5)
    clutch_raw = (ot + (1.0 - min(0.95, pn)) + gi) / 3.0
    if clutch_raw >= 0.58:
        clutch = "high"
    elif clutch_raw <= 0.42:
        clutch = "low"
    else:
        clutch = "medium"

    return (char, mp, cstyle, clutch)


def sync_player_narrative_surface(player: Any, profile: JourneyProfile) -> None:
    """Mirror profile bands onto the player object for engine / logs (idempotent)."""
    profile.narrative_character, profile.media_pressure_band, profile.consistency_style, profile.clutch_factor = (
        _derive_psych_bands(player)
    )
    setattr(player, "personality", profile.narrative_character)
    setattr(player, "media_pressure", profile.media_pressure_band)
    setattr(player, "consistency", profile.consistency_style)
    setattr(player, "clutch_factor", profile.clutch_factor)
    setattr(player, "narrative_state", profile.narrative_state)
    setattr(player, "_narrative_surface_synced", True)


def touch_league_narrative_profiles(league: Any, rng: Optional[random.Random] = None) -> None:
    """Seed psych-derived narrative bands on roster players; safe each season from engine/runner."""
    _ = rng
    for t in getattr(league, "teams", None) or []:
        for p in getattr(t, "roster", None) or []:
            char, mp, cs, cf = _derive_psych_bands(p)
            setattr(p, "personality", char)
            setattr(p, "media_pressure", mp)
            setattr(p, "consistency", cs)
            setattr(p, "clutch_factor", cf)
            if not getattr(p, "narrative_state", None):
                setattr(p, "narrative_state", STATE_STABLE_PRO)


def _narrative_flavor_suffix(profile: JourneyProfile, negative: bool) -> str:
    if profile.narrative_character == "low" and negative:
        return " Off-ice noise and inconsistency have followed him."
    if profile.narrative_character == "high" and not negative:
        return " Teammates lean on his steadiness when games tighten."
    if profile.media_pressure_band == "high" and negative:
        return " The spotlight has been unforgiving."
    return ""


def _compute_narrative_state(
    ovr: float,
    prev_ovr: float,
    age: int,
    league_avg: float,
    profile: JourneyProfile,
) -> str:
    delta = ovr - prev_ovr if prev_ovr > 0 else 0.0
    if ovr >= 0.90 and profile.seasons_played >= 2:
        return STATE_SUPERSTAR
    if delta >= 0.06 and ovr >= 0.72:
        return STATE_BREAKOUT_STAR
    if ovr >= 0.78 and age <= 26:
        return STATE_RISING_STAR
    if prev_ovr > 0 and delta <= -0.05 and age >= 29:
        return STATE_DECLINING_VETERAN
    if prev_ovr > 0 and delta <= -0.04:
        return STATE_STRUGGLING
    if ovr < league_avg - 0.06 and ovr < 0.68:
        return STATE_UNDERDOG
    return STATE_STABLE_PRO


def narrative_tick_active_events(
    profiles: Dict[str, JourneyProfile], year: int, context: Dict[str, Any]
) -> None:
    """Decrement durations; emit resolution lines into context."""
    res: List[str] = []
    for prof in profiles.values():
        kept: List[Dict[str, Any]] = []
        for ev in prof.narrative_active_events:
            try:
                rs = int(ev.get("remaining_seasons", 0) or 0) - 1
            except (TypeError, ValueError):
                rs = 0
            if rs <= 0:
                res.append(
                    f"{prof.name}: {ev.get('kind', '?')} resolved (end season {year}, "
                    f"had intensity {float(ev.get('intensity', 0) or 0):.2f})"
                )
                prof.recent_narrative_signal_logs.append(
                    {"y": int(year), "op": "~", "kind": str(ev.get("kind") or "?")}
                )
                if len(prof.recent_narrative_signal_logs) > 16:
                    prof.recent_narrative_signal_logs = prof.recent_narrative_signal_logs[-16:]
            else:
                ev["remaining_seasons"] = rs
                kept.append(ev)
        prof.narrative_active_events = kept
    if res:
        context["narrative_resolution_lines"] = res[-48:]
    else:
        context["narrative_resolution_lines"] = []


def _narrative_active_kind_set(profile: JourneyProfile) -> set:
    return {str(ev.get("kind") or "") for ev in profile.narrative_active_events if ev.get("kind")}


def try_activate_narrative_event(
    profile: JourneyProfile,
    year: int,
    kind: str,
    category: str,
    intensity: float,
    duration_seasons: int,
    context: Dict[str, Any],
) -> bool:
    if len(profile.narrative_active_events) >= NARRATIVE_MAX_ACTIVE_EVENTS:
        return False
    if kind in _narrative_active_kind_set(profile):
        return False
    ly = profile.narrative_fired_major_years.get(kind)
    if ly is not None and year - int(ly) < NARRATIVE_MAJOR_COOLDOWN_YEARS:
        return False
    profile.narrative_active_events.append(
        {
            "kind": kind,
            "category": category,
            "intensity": max(0.0, min(1.0, float(intensity))),
            "remaining_seasons": max(1, int(duration_seasons)),
            "started_year": int(year),
        }
    )
    profile.narrative_fired_major_years[kind] = int(year)
    profile.recent_narrative_signal_logs.append({"y": int(year), "op": "+", "kind": kind})
    if len(profile.recent_narrative_signal_logs) > 16:
        profile.recent_narrative_signal_logs = profile.recent_narrative_signal_logs[-16:]
    log = context.setdefault("narrative_activation_lines", [])
    log.append(
        f"NARRATIVE EVENT: {profile.name} -> {kind} [{category}] "
        f"intensity={float(intensity):.2f} duration={int(duration_seasons)}y"
    )
    if len(log) > 200:
        log[:] = log[-200:]
    return True


def narrative_evaluate_mechanical_triggers(
    player: Any,
    _team: Any,
    profile: JourneyProfile,
    year: int,
    ovr: float,
    prev_ovr: float,
    age: int,
    league_avg: float,
    league_median: float,
    rng: random.Random,
    context: Dict[str, Any],
) -> None:
    """Logic-based activation (not spam): ties psych bands, OVR deltas, career stage."""
    if prev_ovr <= 0:
        return
    drop = float(prev_ovr - ovr)
    rise = float(ovr - prev_ovr)
    char = profile.narrative_character
    mp = profile.media_pressure_band
    cstyle = profile.consistency_style
    clutch = profile.clutch_factor
    inten = lambda: float(rng.uniform(0.58, 0.92))

    struggling = profile.narrative_state == STATE_STRUGGLING
    if (
        drop >= 0.038
        and age <= 31
        and (char == "low" or mp == "high")
        and (struggling or drop >= 0.048)
    ):
        try_activate_narrative_event(
            profile, year, _EVT_CONFIDENCE_COLLAPSE, "mental", inten(), 2, context
        )

    if rise >= 0.042 and ovr >= 0.68 and (clutch == "high" or cstyle in ("volatile", "streaky")):
        try_activate_narrative_event(profile, year, _EVT_MOMENTUM_HOT, "performance", inten(), 2, context)

    if mp == "high" and drop >= 0.026 and ovr >= 0.74:
        try_activate_narrative_event(profile, year, _EVT_MEDIA_STRAIN, "off_ice", inten(), 2, context)

    if profile.decline_seasons >= 2 and age >= 27 and mp == "high":
        try_activate_narrative_event(profile, year, _EVT_BURNOUT_WEAR, "mental", inten(), 2, context)

    if profile.narrative_state == STATE_BREAKOUT_STAR and profile.seasons_played <= 5 and ovr >= 0.74:
        try_activate_narrative_event(
            profile, year, _EVT_BREAKOUT_PRESSURE, "career_arc", inten(), 2, context
        )

    if char == "high" and profile.narrative_state in (STATE_RISING_STAR, STATE_BREAKOUT_STAR) and rise >= 0.035:
        try_activate_narrative_event(
            profile, year, _EVT_LEADERSHIP_SURGE, "career_arc", inten(), 2, context
        )

    if TAG_CAREER_RESURGENCE in profile.narrative_tags and rise >= 0.055:
        try_activate_narrative_event(
            profile, year, _EVT_RESURGENCE_DRIVE, "career_arc", inten(), 3, context
        )


def apply_narrative_mechanics_to_rosters(
    league: Any,
    context: Dict[str, Any],
    year: int,
    rng: Optional[random.Random] = None,
    *,
    max_trace_lines: int = 32,
) -> None:
    """
    Before progression each season: aggregate active narrative events into per-player
    modifier attributes consumed by development / regression / engine special rolls.
    """
    r = rng if isinstance(rng, random.Random) else random.Random()
    _ = r
    profiles: Dict[str, JourneyProfile] = context.get("player_journeys") or {}
    trace: List[str] = []
    for team in getattr(league, "teams", None) or []:
        for player in getattr(team, "roster", None) or []:
            if getattr(player, "retired", False):
                continue
            key = _player_key(player)
            prof = profiles.get(key)
            pg = 1.0
            rr = 1.0
            bp = 1.0
            dp = 1.0
            csum = 0.0
            vsum = 0.0
            ev_summary: List[str] = []
            if prof and prof.narrative_active_events:
                for ev in prof.narrative_active_events:
                    d = _narrative_event_effect_deltas(
                        str(ev.get("kind") or ""),
                        float(ev.get("intensity", 0.7) or 0.7),
                    )
                    pg *= float(d["prog_growth"])
                    rr *= float(d["regression_rate"])
                    bp *= float(d["breakout_p"])
                    dp *= float(d["decline_p"])
                    csum += float(d["consistency"])
                    vsum += float(d["variance"])
                    ev_summary.append(str(ev.get("kind")))
            pg = max(0.78, min(1.18, pg))
            rr = max(0.74, min(1.30, rr))
            bp = max(0.68, min(1.22, bp))
            dp = max(0.72, min(1.32, dp))
            csum = max(-0.22, min(0.18, csum))
            vsum = max(0.0, min(0.22, vsum))

            setattr(player, "_narrative_prog_growth_mult", pg)
            setattr(player, "_narrative_regression_rate_mult", rr)
            setattr(player, "_narrative_breakout_p_mult", bp)
            setattr(player, "_narrative_decline_p_mult", dp)
            setattr(player, "_narrative_consistency_shift", csum)
            setattr(player, "_narrative_performance_variance", vsum)
            setattr(player, "_narrative_mechanics_year", int(year))

            psych = getattr(player, "psych", None)
            if psych is not None:
                try:
                    base_c = float(getattr(psych, "confidence_level", 0.5) or 0.5)
                    psych.confidence_level = max(
                        0.08, min(0.96, base_c + 0.14 * csum - 0.06 * vsum)
                    )
                    if hasattr(psych, "clamp_all"):
                        psych.clamp_all()
                except Exception:
                    pass

            if ev_summary and len(trace) < max_trace_lines:
                nm = _player_name(player)
                trace.append(
                    f"EFFECT TRACE: {nm} active={','.join(ev_summary)} "
                    f"prog_x={pg:.3f} reg_x={rr:.3f} breakout_p_x={bp:.3f} decline_p_x={dp:.3f} "
                    f"cons_shift={csum:+.3f} var={vsum:.3f}"
                )

    context["narrative_effect_trace_lines"] = trace


def build_player_journey_digest_lines(
    context: Dict[str, Any],
    league: Any,
    year: int,
    max_lines: int = 10,
) -> List[str]:
    """
    Multi-sentence season digest for standout careers (uses profile memory, not raw spam).
    """
    profiles: Dict[str, JourneyProfile] = context.get("player_journeys") or {}
    if not profiles:
        return []

    def score_prof(prof: JourneyProfile) -> float:
        peak = float(prof.peak_ovr)
        seas = int(prof.seasons_played)
        tag_w = 0.0
        for tg in prof.narrative_tags:
            if tg in (TAG_SUPERSTAR, TAG_GENERATIONAL_TALENT, TAG_ROOKIE_SENSATION):
                tag_w += 0.35
            if tg in (TAG_BREAKOUT, TAG_LATE_BLOOMER, TAG_CAREER_RESURGENCE):
                tag_w += 0.22
            if tg in (TAG_DECLINING_STAR, TAG_FALL_FROM_GRACE, TAG_REPEATED_DECLINE):
                tag_w += 0.18
        return peak * 1.1 + min(6, seas) * 0.04 + tag_w

    ranked = sorted(profiles.values(), key=score_prof, reverse=True)[: max_lines * 2]
    lines: List[str] = []
    seen: set[str] = set()
    for prof in ranked:
        if len(lines) >= max_lines:
            break
        if prof.player_key in seen:
            continue
        seen.add(prof.player_key)
        age = year - prof.birth_year if prof.birth_year else 0
        ctx = ""
        for t in getattr(league, "teams", None) or []:
            for p in getattr(t, "roster", None) or []:
                if _player_key(p) == prof.player_key:
                    ctx = _team_context_phrase(t)
                    break
            if ctx:
                break
        if not ctx:
            ctx = "his team"

        parts: List[str] = []
        if prof.narrative_state == STATE_SUPERSTAR:
            parts.append(
                f"At {age}, {prof.name} is appointment viewing - the engine of {ctx}."
            )
        elif prof.narrative_state == STATE_BREAKOUT_STAR:
            parts.append(
                f"{prof.name} ({age}) took another leap; the league is adjusting to his timing."
            )
        elif prof.narrative_state == STATE_DECLINING_VETERAN:
            parts.append(
                f"The miles are showing for {prof.name} ({age}), and {ctx} is weighing how much runway remains."
            )
        elif prof.narrative_state == STATE_STRUGGLING:
            parts.append(
                f"{prof.name} is fighting the scoreboard this year; patience in {ctx} is being tested."
            )
        elif TAG_CAREER_RESURGENCE in prof.narrative_tags:
            parts.append(f"{prof.name} is authoring a redemption chapter after seasons of doubt.")
        elif TAG_FALL_FROM_GRACE in prof.narrative_tags:
            parts.append(
                f"Once untouchable, {prof.name} is learning how fast the league moves on from icons."
            )
        elif prof.former_team_ids and prof.seasons_played >= 2:
            parts.append(
                f"{prof.name} is still settling into a new chapter after changing addresses mid-career."
            )
        elif TAG_UNDERDOG_RISE in prof.narrative_tags or prof.narrative_state == STATE_UNDERDOG:
            parts.append(
                f"{prof.name} keeps carving NHL minutes the hard way - proof that {ctx} values compete level."
            )
        else:
            if prof.peak_ovr >= 0.82:
                parts.append(
                    f"{prof.name} remains a high-trust piece for {ctx}, still squarely in his prime years."
                )
            else:
                continue
        if parts:
            lines.append(parts[0])
    return lines


def _get_all_roster_ovrs(league: Any) -> List[Tuple[Any, Any, float, int]]:
    """(player, team, ovr, age) for every roster player."""
    out: List[Tuple[Any, Any, float, int]] = []
    for t in getattr(league, "teams", None) or []:
        for p in getattr(t, "roster", None) or []:
            ovr_fn = getattr(p, "ovr", None)
            ovr = float(ovr_fn()) if callable(ovr_fn) else float(getattr(p, "ovr", 0.5))
            ident = getattr(p, "identity", None)
            age = int(getattr(ident, "age", 26) or 26)
            out.append((p, t, ovr, age))
    return out


def update_player_journeys(
    league: Any,
    year: int,
    context: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Update journey profiles from current rosters and emit narrative events.
    context["player_journeys"] = dict[player_key -> JourneyProfile]
    Returns list of {"type": "player_journey", "player_name", "team_name", "message", "tag"}.
    """
    events: List[Dict[str, Any]] = []
    profiles: Dict[str, JourneyProfile] = context.get("player_journeys") or {}
    context["player_journeys"] = profiles
    context["narrative_activation_lines"] = []
    journal_rng = rng if isinstance(rng, random.Random) else random.Random()
    narrative_tick_active_events(profiles, year, context)

    all_ovrs = _get_all_roster_ovrs(league)
    if not all_ovrs:
        return events

    ovrs_only = [x[2] for x in all_ovrs]
    league_avg_ovr = sum(ovrs_only) / len(ovrs_only)
    league_median_ovr = sorted(ovrs_only)[len(ovrs_only) // 2] if ovrs_only else 0.55
    n_players = len(all_ovrs)
    top_10_pct_ovr = sorted(ovrs_only, reverse=True)[max(0, n_players // 10)] if n_players >= 10 else 0.90

    for player, team, ovr, age in all_ovrs:
        key = _player_key(player)
        name = _player_name(player)
        tname = _team_name(team)
        tid = _team_id(team)
        profile = profiles.get(key)
        if profile is None:
            profile = JourneyProfile(
                player_key=key,
                name=name,
                birth_year=year - age,
            )
            profiles[key] = profile

        prev_ovr = profile.prev_ovr
        is_rookie = profile.seasons_played == 0
        tc = _team_context_phrase(team)

        if profile.last_team_id and profile.last_team_id != tid and profile.seasons_played > 0:
            if profile.last_team_id not in profile.former_team_ids:
                profile.former_team_ids.append(profile.last_team_id)
            _append_major_event(profile, year, "team_change", tid)
            events.append({
                "type": "player_journey",
                "player_name": name,
                "team_name": tname,
                "message": (
                    f"{name} opens a new chapter in {tname}, chasing a bigger role inside {tc}."
                ),
                "tag": "team_change",
            })

        profile.seasons_played += 1
        profile.prev_prev_ovr = profile.prev_ovr
        profile.prev_ovr = ovr
        if ovr > profile.peak_ovr:
            profile.peak_ovr = ovr
        if is_rookie:
            profile.last_rookie_year = year

        # --- Rookie breakout: first season, well above league average
        if is_rookie and ovr >= league_avg_ovr + 0.12 and ovr >= 0.75:
            if TAG_ROOKIE_SENSATION not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_ROOKIE_SENSATION)
                msg = (
                    f"{name} explodes onto the scene with a rookie year that reframes what {tc} can become."
                )
                msg += _narrative_flavor_suffix(profile, False)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": msg,
                    "tag": TAG_ROOKIE_SENSATION,
                })
                _append_major_event(profile, year, "rookie_sensation", "")

        # --- Breakout + repeat breakout arcs
        if (
            profile.seasons_played >= 2
            and prev_ovr > 0
            and (ovr - prev_ovr) >= 0.07
            and ovr >= 0.72
            and TAG_ROOKIE_SENSATION not in profile.narrative_tags
        ):
            if TAG_BREAKOUT not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_BREAKOUT)
                profile.breakout_seasons += 1
                msg = (
                    f"After years of quiet development, {name} breaks out as a top-line driver for {tc}."
                )
                msg += _narrative_flavor_suffix(profile, False)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": msg,
                    "tag": TAG_BREAKOUT,
                })
                _append_major_event(profile, year, "breakout", "")
            elif TAG_BREAKOUT_REPEAT not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_BREAKOUT_REPEAT)
                profile.breakout_seasons += 1
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": (
                        f"This is no fluke anymore: {name} stacks another star-level season and forces "
                        f"the league to respect {tc}."
                    ),
                    "tag": TAG_BREAKOUT_REPEAT,
                })
                _append_major_event(profile, year, "breakout_repeat", "")
        elif (
            TAG_LATE_BLOOMER not in profile.narrative_tags
            and TAG_BREAKOUT not in profile.narrative_tags
            and age >= 23
            and profile.seasons_played >= 2
            and prev_ovr > 0
            and (ovr - prev_ovr) >= 0.08
            and ovr >= 0.70
            and TAG_ROOKIE_SENSATION not in profile.narrative_tags
        ):
            profile.narrative_tags.append(TAG_LATE_BLOOMER)
            msg = (
                f"Years of uneven hockey behind him, {name} finally puts it together - a true late bloomer for {tc}."
            )
            msg += _narrative_flavor_suffix(profile, False)
            events.append({
                "type": "player_journey",
                "player_name": name,
                "team_name": tname,
                "message": msg,
                "tag": TAG_LATE_BLOOMER,
            })

        # --- Superstar: elite OVR sustained
        if ovr >= 0.90 and profile.seasons_played >= 2 and prev_ovr >= 0.88:
            if TAG_SUPERSTAR not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_SUPERSTAR)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": (
                        f"{name} is no longer rising - he's arrived, carrying {tc} like a franchise pillar."
                    ),
                    "tag": TAG_SUPERSTAR,
                })

        # --- Generational: top-of-league sustained
        if ovr >= 0.92 and profile.peak_ovr >= 0.92 and profile.seasons_played >= 3:
            if TAG_GENERATIONAL_TALENT not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_GENERATIONAL_TALENT)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": f"{name} steps into generational territory - the kind of talent eras orbit around.",
                    "tag": TAG_GENERATIONAL_TALENT,
                })

        # --- Decline + repeated decline
        drop = (prev_ovr - ovr) if prev_ovr > 0 else 0.0
        if age >= 28 and prev_ovr > 0 and drop >= 0.045 and ovr < prev_ovr:
            profile.decline_seasons += 1
        elif prev_ovr > 0 and ovr > prev_ovr + 0.02:
            profile.decline_seasons = max(0, profile.decline_seasons - 1)

        if age >= 30 and prev_ovr > 0 and drop >= 0.05 and ovr < prev_ovr:
            profile.decline_indicator = True
            if TAG_DECLINING_STAR not in profile.narrative_tags and prev_ovr >= 0.85:
                profile.narrative_tags.append(TAG_DECLINING_STAR)
                profile.was_declining = True
                msg = (
                    f"Regression shows up in the details for {name}: age is winning the race against reputation."
                )
                msg += _narrative_flavor_suffix(profile, True)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": msg,
                    "tag": TAG_DECLINING_STAR,
                })
                _append_major_event(profile, year, "decline", "")
            elif (
                TAG_REPEATED_DECLINE not in profile.narrative_tags
                and TAG_DECLINING_STAR in profile.narrative_tags
                and profile.decline_seasons >= 2
                and drop >= 0.04
            ):
                profile.narrative_tags.append(TAG_REPEATED_DECLINE)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": (
                        f"What started as whispers is now loud: {name}, once automatic, is fighting the clock every shift."
                    ),
                    "tag": TAG_REPEATED_DECLINE,
                })

        # --- Fall from grace (elite peak, sharp slide)
        if (
            TAG_FALL_FROM_GRACE not in profile.narrative_tags
            and (TAG_SUPERSTAR in profile.narrative_tags or TAG_GENERATIONAL_TALENT in profile.narrative_tags)
            and prev_ovr >= 0.88
            and ovr <= 0.82
            and drop >= 0.06
        ):
            profile.narrative_tags.append(TAG_FALL_FROM_GRACE)
            events.append({
                "type": "player_journey",
                "player_name": name,
                "team_name": tname,
                "message": (
                    f"The league built game plans around {name}; now those same coaches are asking if the magic is gone."
                ),
                "tag": TAG_FALL_FROM_GRACE,
            })

        # --- Resurgence: veteran improving after decline
        if age >= 28 and profile.was_declining and prev_ovr > 0 and (ovr - prev_ovr) >= 0.06:
            if TAG_CAREER_RESURGENCE not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_CAREER_RESURGENCE)
                msg = (
                    f"Written off too soon? {name} answers back with a season that smells like redemption for {tc}."
                )
                msg += _narrative_flavor_suffix(profile, False)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": msg,
                    "tag": TAG_CAREER_RESURGENCE,
                })
                profile.was_declining = False

        # --- Volatile star (personality + big swing)
        char0, _, cstyle, _ = _derive_psych_bands(player)
        profile.narrative_character = char0
        if (
            TAG_VOLATILE_STAR not in profile.narrative_tags
            and cstyle == "volatile"
            and ovr >= 0.78
            and prev_ovr > 0
            and abs(ovr - prev_ovr) >= 0.055
        ):
            profile.narrative_tags.append(TAG_VOLATILE_STAR)
            events.append({
                "type": "player_journey",
                "player_name": name,
                "team_name": tname,
                "message": (
                    f"{name} remains box-office - brilliant one week, baffling the next - and {tc} lives with the swings."
                ),
                "tag": TAG_VOLATILE_STAR,
            })

        # --- Underdog rise
        if (
            TAG_UNDERDOG_RISE not in profile.narrative_tags
            and age <= 26
            and prev_ovr > 0
            and ovr < league_avg_ovr - 0.04
            and (ovr - prev_ovr) >= 0.05
        ):
            profile.narrative_tags.append(TAG_UNDERDOG_RISE)
            events.append({
                "type": "player_journey",
                "player_name": name,
                "team_name": tname,
                "message": (
                    f"Undersized on the marquee, {name} earns his keep anyway - the kind of story {tc} loves to tell."
                ),
                "tag": TAG_UNDERDOG_RISE,
            })

        # --- Bust: young, was expected (high draft) but consistently low OVR
        if age <= 25 and profile.seasons_played >= 3 and ovr <= league_median_ovr - 0.08 and profile.peak_ovr < 0.72:
            if TAG_BUST not in profile.narrative_tags:
                profile.narrative_tags.append(TAG_BUST)
                msg = f"The projection never matched the production for {name} - a sobering arc for {tc}."
                msg += _narrative_flavor_suffix(profile, True)
                events.append({
                    "type": "player_journey",
                    "player_name": name,
                    "team_name": tname,
                    "message": msg,
                    "tag": TAG_BUST,
                })

        profile.narrative_state = _compute_narrative_state(ovr, prev_ovr, age, league_avg_ovr, profile)
        narrative_evaluate_mechanical_triggers(
            player,
            team,
            profile,
            year,
            ovr,
            prev_ovr,
            age,
            league_avg_ovr,
            league_median_ovr,
            journal_rng,
            context,
        )
        profile.ovr_history.append((year, round(ovr, 4)))
        if len(profile.ovr_history) > 14:
            profile.ovr_history.pop(0)
        sync_player_narrative_surface(player, profile)
        profile.last_team_id = tid

    return events
