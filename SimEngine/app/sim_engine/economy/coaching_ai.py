# backend/app/sim_engine/economy/waiver_ai.py
"""
Waiver Priority + Claim AI — NHL Franchise Mode Sim (Economy)

Goals:
- NHL-authentic waiver priority:
  - Early season: reverse order of previous season standings
  - After cutoff: reverse order of current season standings
  - After successful claim: claiming team moves to the back (rolling priority)
- Deterministic (no unseeded randomness)
- Standard library only
- No I/O, no HTTP, no DB

This module is intentionally "integration-friendly":
- It accepts standings/team/player data as dicts.
- It doesn't assume your exact League/Season classes; it just needs data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# Types
# =============================================================================

TeamDict = Dict[str, Any]
PlayerDict = Dict[str, Any]


# =============================================================================
# Helpers
# =============================================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _norm_pos(pos: Any) -> str:
    if pos is None:
        return ""
    p = str(pos).strip().upper()
    # normalize common forms
    if p in ("LW", "RW", "W"):
        return "W"
    if p in ("C", "CENTER"):
        return "C"
    if p in ("D", "DEF", "DEFENSE", "DEFENCEMAN"):
        return "D"
    if p in ("G", "GOALIE", "GK"):
        return "G"
    return p


def _team_id(team: TeamDict) -> str:
    tid = team.get("team_id")
    if tid is None:
        raise ValueError("Team dict missing required key: 'team_id'")
    return str(tid)


def _player_pos(player: PlayerDict) -> str:
    return _norm_pos(player.get("position", ""))


def _team_window(team: TeamDict) -> str:
    # expected: rebuild | bubble | contender | tank
    return str(team.get("competitive_window", "bubble")).strip().lower()


def _has_roster_slot(team: TeamDict) -> bool:
    # If your engine tracks it, honor it. Otherwise assume slot exists.
    # Supported keys: roster_open_slots, roster_full, roster_size, roster_limit
    if "roster_open_slots" in team:
        return _safe_int(team.get("roster_open_slots"), 0) > 0
    if "roster_full" in team:
        return not bool(team.get("roster_full"))
    if "roster_size" in team and "roster_limit" in team:
        return _safe_int(team["roster_size"]) < _safe_int(team["roster_limit"])
    return True


def _can_afford(team: TeamDict, player: PlayerDict) -> bool:
    cap_space = _safe_float(team.get("cap_space"), 0.0)
    cap_hit = _safe_float(player.get("cap_hit"), 0.0)
    # If your sim has "budget room" etc, extend here later.
    return cap_space >= cap_hit


def _need_match_bonus(team: TeamDict, player: PlayerDict) -> float:
    needs = team.get("roster_needs", []) or []
    needs_norm = {_norm_pos(n) for n in needs}
    pos = _player_pos(player)
    if not pos:
        return 0.0
    return 1.0 if pos in needs_norm else 0.0


def _upgrade_score(team: TeamDict, player: PlayerDict) -> float:
    """
    Optional upgrade estimation:
    - if you pass team['replacement_level'][pos] or team['avg_ovr_by_pos'][pos], we use it
    - otherwise we treat player overall_projection as-is
    """
    pos = _player_pos(player)
    p_ovr = _safe_float(player.get("overall_projection"), 0.0)

    # Supported optional structures
    rep = team.get("replacement_level", {}) or {}
    avg = team.get("avg_ovr_by_pos", {}) or {}

    baseline = None
    if pos and pos in rep:
        baseline = _safe_float(rep.get(pos), 0.0)
    elif pos and pos in avg:
        baseline = _safe_float(avg.get(pos), 0.0)

    if baseline is None:
        # No baseline info; assume raw projection provides some value
        return _clamp(p_ovr, 0.0, 1.0)

    # Improvement over baseline, scaled
    return _clamp(p_ovr - baseline, -1.0, 1.0)


def _contract_risk_penalty(team: TeamDict, player: PlayerDict) -> float:
    """
    Penalize long expensive deals for tank/rebuild teams.
    """
    years = _safe_int(player.get("contract_years_left"), 0)
    cap_hit = _safe_float(player.get("cap_hit"), 0.0)
    window = _team_window(team)

    # Simple, deterministic penalties:
    # - long years hurts tank/rebuild more
    # - high cap hit hurts everyone a bit
    years_pen = 0.0
    if years >= 3:
        years_pen = 0.20 if window in ("tank", "rebuild") else 0.10
    if years >= 5:
        years_pen += 0.15 if window in ("tank", "rebuild") else 0.05

    cap_pen = _clamp(cap_hit / 12.0, 0.0, 0.35)  # 12M ~ max-ish cap hit in modern NHL

    return _clamp(years_pen + cap_pen, 0.0, 0.75)


def _age_fit_bonus(team: TeamDict, player: PlayerDict) -> float:
    """
    Deterministic age curve preference by window:
    - tank/rebuild: prefer younger
    - contender: prefer prime
    - bubble: mild preference for prime/younger
    """
    age = _safe_int(player.get("age"), 25)
    window = _team_window(team)

    # crude curve
    if window in ("tank", "rebuild"):
        if age <= 21:
            return 0.20
        if age <= 24:
            return 0.10
        if age <= 28:
            return 0.02
        return -0.08

    if window == "contender":
        if 24 <= age <= 30:
            return 0.15
        if 31 <= age <= 34:
            return 0.08
        if age <= 23:
            return 0.03
        return -0.05

    # bubble/default
    if age <= 23:
        return 0.08
    if 24 <= age <= 30:
        return 0.10
    if 31 <= age <= 34:
        return 0.04
    return -0.05


def _window_aggression(window: str) -> float:
    """
    How eager the archetype is to claim a waiver guy.
    Bigger = more likely to claim.
    """
    w = window.strip().lower()
    if w == "tank":
        return 0.35
    if w == "rebuild":
        return 0.30
    if w == "bubble":
        return 0.22
    if w == "contender":
        return 0.18
    return 0.22


def _claim_threshold(window: str) -> float:
    """
    Minimum score to actually place a claim.
    Lower threshold = more claims.
    """
    w = window.strip().lower()
    if w == "tank":
        return 0.38
    if w == "rebuild":
        return 0.40
    if w == "bubble":
        return 0.45
    if w == "contender":
        return 0.50
    return 0.45


# =============================================================================
# Standings sorting (reverse order = worst-first)
# =============================================================================

def _standings_sort_key(team: TeamDict) -> Tuple[float, float, float, str]:
    """
    Worst-first ordering:
    1) points ascending
    2) point_pct ascending (fallback if points tie)
    3) goal_diff ascending (worse differential first)
    4) team_id lexicographic (deterministic final tie-break)
    """
    pts = _safe_int(team.get("points"), 0)
    pct = _safe_float(team.get("point_pct"), 0.0)
    gd = _safe_int(team.get("goal_diff"), 0)
    return (float(pts), float(pct), float(gd), _team_id(team))


def build_waiver_priority_from_standings(standings: Sequence[TeamDict]) -> List[str]:
    """
    Given standings list, return waiver priority list worst-first.

    standings: list of TeamDicts containing at least 'team_id' and ideally points/point_pct/goal_diff.
    """
    # sort worst-first (ascending by points, pct, gd)
    ordered = sorted(list(standings), key=_standings_sort_key)
    return [_team_id(t) for t in ordered]


def update_priority_after_claim(priority_order: Sequence[str], claiming_team_id: str) -> List[str]:
    """
    Rolling priority: claiming team moves to the end.
    """
    cid = str(claiming_team_id)
    new_order = [tid for tid in priority_order if tid != cid]
    new_order.append(cid)
    return new_order


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class WaiverConfig:
    # "Early season uses previous year standings" cutoff:
    # You can map this to "season day" or "sim day" in your SeasonContext.
    early_season_cutoff_day: int = 30


# =============================================================================
# Engine
# =============================================================================

class WaiverEngine:
    """
    Core waiver engine.

    You can use this in two ways:

    A) High-level (preferred):
        engine = WaiverEngine(config=WaiverConfig(...))
        order = engine.build_priority(league_context, season_context)
        winner = engine.process_player(player, teams, order)

    B) Low-level:
        order = build_waiver_priority_from_standings(...)
        winner = engine.process_player(player, teams, order)
    """

    def __init__(self, config: WaiverConfig = WaiverConfig()):
        self.config = config

    # -----------------------------
    # Priority building
    # -----------------------------

    def build_priority(self, league_context: Any, season_context: Any) -> List[str]:
        """
        Determine correct waiver priority order:
        - if early season: use previous season standings
        - else: use current standings

        Expected data (flexible):
        - season_context may provide:
            - season_context["day"] or season_context.day
            - season_context["standings_current"] or season_context.standings_current
        - league_context may provide:
            - league_context["standings_prev"] or league_context.standings_prev

        Standings format: List[TeamDict]
        """
        day = self._get_value(season_context, "day", default=0)
        day_i = _safe_int(day, 0)

        if day_i <= self.config.early_season_cutoff_day:
            prev = self._get_value(league_context, "standings_prev", default=None)
            if prev is None:
                # fallback: if prev missing, use current
                cur = self._get_value(season_context, "standings_current", default=[])
                return build_waiver_priority_from_standings(cur)
            return build_waiver_priority_from_standings(prev)

        cur = self._get_value(season_context, "standings_current", default=[])
        return build_waiver_priority_from_standings(cur)

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        # dict-like
        if isinstance(obj, dict):
            return obj.get(key, default)
        # attribute-like
        return getattr(obj, key, default)

    # -----------------------------
    # Claim processing
    # -----------------------------

    def process_player(
        self,
        player: PlayerDict,
        teams: Sequence[TeamDict],
        priority_order: Sequence[str],
    ) -> Optional[str]:
        """
        Iterate teams in priority order:
        - Evaluate whether each team places a claim
        - First claimant wins
        - Return winning team_id or None (clears waivers)

        NOTE:
        - This function does NOT mutate your roster/cap tables.
        - It only decides the winning claim.
        - Use the returned team_id to perform actual roster move elsewhere.
        """
        teams_by_id = {_team_id(t): t for t in teams}

        for tid in priority_order:
            team = teams_by_id.get(str(tid))
            if team is None:
                continue  # team not in this claim run

            if not _has_roster_slot(team):
                continue

            if not _can_afford(team, player):
                continue

            if self._team_wants_to_claim(team, player):
                return _team_id(team)

        return None

    def process_player_and_update_priority(
        self,
        player: PlayerDict,
        teams: Sequence[TeamDict],
        priority_order: Sequence[str],
    ) -> Tuple[Optional[str], List[str]]:
        """
        Convenience helper:
        - Returns (winning_team_id, new_priority_order)
        """
        winner = self.process_player(player, teams, priority_order)
        if winner is None:
            return None, list(priority_order)
        return winner, update_priority_after_claim(priority_order, winner)

    # -----------------------------
    # Decision model
    # -----------------------------

    def _team_wants_to_claim(self, team: TeamDict, player: PlayerDict) -> bool:
        """
        Deterministic scoring model — no randomness.
        """
        window = _team_window(team)
        aggression = _window_aggression(window)
        threshold = _claim_threshold(window)

        # Position need
        need_bonus = 0.20 * _need_match_bonus(team, player)

        # Cap affordability (extra reward if lots of space)
        cap_space = _safe_float(team.get("cap_space"), 0.0)
        cap_hit = _safe_float(player.get("cap_hit"), 0.0)
        # If cap_hit == 0, treat as very affordable
        affordability = 0.0
        if cap_hit <= 0.001:
            affordability = 0.15
        else:
            ratio = cap_space / max(cap_hit, 0.1)
            # ratio 1.0 means barely fits; 2.0 comfy; 4.0 very comfy
            affordability = _clamp((ratio - 1.0) * 0.08, -0.10, 0.20)

        # Talent / upgrade
        upgrade = _upgrade_score(team, player)
        upgrade_bonus = 0.30 * _clamp(upgrade, 0.0, 1.0)

        # Age fit
        age_fit = _age_fit_bonus(team, player)  # already signed-ish
        age_bonus = age_fit

        # Contract risk penalty
        risk_pen = _contract_risk_penalty(team, player)

        # Window-based appetite
        window_bonus = aggression

        # Final score
        score = window_bonus + need_bonus + affordability + upgrade_bonus + age_bonus - risk_pen

        # Optional knobs (team personality / GM style)
        # If you pass team["waiver_bias"] in [-0.2, +0.2], it adjusts deterministically.
        bias = _safe_float(team.get("waiver_bias"), 0.0)
        score += _clamp(bias, -0.20, 0.20)

        return score >= threshold


# =============================================================================
# Minimal “pure functions” API (if you don’t want the class)
# =============================================================================

def build_waiver_priority(
    league_context: Any,
    season_context: Any,
    early_season_cutoff_day: int = 30,
) -> List[str]:
    """
    Functional wrapper for priority building.
    """
    engine = WaiverEngine(WaiverConfig(early_season_cutoff_day=early_season_cutoff_day))
    return engine.build_priority(league_context, season_context)


def process_waiver_claim(
    player: PlayerDict,
    teams: Sequence[TeamDict],
    priority_order: Sequence[str],
) -> Optional[str]:
    """
    Functional wrapper for claim processing (uses default config/scoring).
    """
    engine = WaiverEngine()
    return engine.process_player(player, teams, priority_order)



# =============================================================================
# Coach tactics / matchup coaching / load management
# =============================================================================
#
# Settings live on the franchise session (``session.team_tactics[tid]``) and are
# stamped onto the team object (``team._tactics``) before each game. The sim reads
# them through ``team_tactic_profile`` (cached per settings + identity) and
# ``game_tactic_mods`` (a handful of multiplications per game), so tactics never
# add per-event work to the light or full paths.

TACTIC_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "line_matching": ("off", "light", "heavy"),
    "goalie_pull": ("conservative", "normal", "aggressive"),
    "forecheck": ("1-2-2", "2-1-2", "2-3"),
    "pace": ("slow", "normal", "fast"),
    "shot_priority": ("quality", "balanced", "volume"),
    "d_activation": ("conservative", "normal", "active"),
    "pp_system": ("umbrella", "1-3-1", "overload"),
    "pk_system": ("box", "diamond", "pressure"),
    "b2b_goalie": ("auto", "starter", "backup"),
}

DEFAULT_TACTICS: Dict[str, Any] = {
    "line_matching": "light",
    "goalie_pull": "normal",
    "forecheck": "2-1-2",
    "pace": "normal",
    "shot_priority": "balanced",
    "d_activation": "normal",
    "pp_system": "1-3-1",
    "pk_system": "box",
    "b2b_goalie": "auto",
    "rest_veterans": False,
    "rest_next": [],
}

# (shot volume, xG per shot, opponent chance allowance, penalties taken, fatigue load)
_TACTIC_EFFECTS: Dict[str, Dict[str, Tuple[float, float, float, float, float]]] = {
    "forecheck": {
        "1-2-2": (-0.030, 0.000, -0.035, -0.06, -0.05),
        "2-1-2": (0.0, 0.0, 0.0, 0.0, 0.0),
        "2-3": (0.040, 0.005, 0.040, 0.10, 0.08),
    },
    "pace": {
        "slow": (-0.040, 0.010, -0.040, -0.03, -0.06),
        "normal": (0.0, 0.0, 0.0, 0.0, 0.0),
        "fast": (0.045, -0.005, 0.045, 0.02, 0.07),
    },
    "shot_priority": {
        "quality": (-0.070, 0.065, 0.0, 0.0, 0.0),
        "balanced": (0.0, 0.0, 0.0, 0.0, 0.0),
        "volume": (0.085, -0.070, 0.0, 0.0, 0.01),
    },
    "d_activation": {
        "conservative": (-0.020, -0.010, -0.030, 0.0, -0.02),
        "normal": (0.0, 0.0, 0.0, 0.0, 0.0),
        "active": (0.030, 0.015, 0.035, 0.02, 0.03),
    },
}

# Which identity profile groups (percentiles 0-100) make each choice work.
_TACTIC_FIT_GROUPS: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("forecheck", "2-3"): ("physical", "pace"),
    ("forecheck", "1-2-2"): ("defense", "goaltending"),
    ("pace", "fast"): ("pace", "puck"),
    ("pace", "slow"): ("defense", "puck"),
    ("shot_priority", "quality"): ("playmaking", "puck"),
    ("shot_priority", "volume"): ("shooting", "physical"),
    ("d_activation", "active"): ("puck", "pace"),
    ("d_activation", "conservative"): ("defense",),
    ("pp_system", "umbrella"): ("shooting",),
    ("pp_system", "1-3-1"): ("playmaking",),
    ("pp_system", "overload"): ("puck", "physical"),
    ("pk_system", "box"): ("defense",),
    ("pk_system", "diamond"): ("defense", "puck"),
    ("pk_system", "pressure"): ("pace", "physical"),
}

_MATCH_LEVEL = {"off": 0.0, "light": 0.5, "heavy": 1.0}
# Late-game pull vs the "normal" baseline: (extra chance the trailing-by-1 club ties it,
# extra empty-net-goal-against chance). Conservative removes some 2-goal empty-netters.
_PULL_DELTA = {"conservative": (0.0, -0.06), "normal": (0.0, 0.0), "aggressive": (0.05, 0.08)}


def normalize_tactics(raw: Any) -> Dict[str, Any]:
    """Validated copy of a tactics dict (unknown keys/values fall back to defaults)."""
    src = raw if isinstance(raw, dict) else {}
    out: Dict[str, Any] = {}
    for k, opts in TACTIC_OPTIONS.items():
        v = str(src.get(k, DEFAULT_TACTICS[k]) or DEFAULT_TACTICS[k])
        out[k] = v if v in opts else DEFAULT_TACTICS[k]
    out["rest_veterans"] = bool(src.get("rest_veterans", False))
    rn = src.get("rest_next") or []
    out["rest_next"] = [str(x) for x in rn if x is not None][:6] if isinstance(rn, (list, tuple, set)) else []
    return out


def _identity_pct(team: Any) -> Dict[str, float]:
    ident = getattr(team, "team_identity", None)
    if isinstance(ident, dict):
        pct = ident.get("profile_pct")
        if isinstance(pct, dict):
            return {str(k): _safe_float(v, 50.0) for k, v in pct.items()}
    return {}


def cpu_tactics_for_team(team: Any) -> Dict[str, Any]:
    """CPU coach picks a system from the club's identity (falls back to defaults)."""
    t = dict(DEFAULT_TACTICS)
    t["rest_next"] = []
    ident = getattr(team, "team_identity", None)
    key = ""
    if isinstance(ident, dict):
        key = str((ident.get("primary") or {}).get("key") or "")
    pct = _identity_pct(team)
    if key == "heavy_forecheck":
        t.update(forecheck="2-3", shot_priority="volume", pk_system="pressure", pp_system="overload")
    elif key in ("run_and_gun", "speed_transition"):
        t.update(pace="fast", d_activation="active", pp_system="1-3-1", pk_system="diamond")
    elif key in ("defensive_structure", "goaltending_fortress"):
        t.update(forecheck="1-2-2", pace="slow", d_activation="conservative", shot_priority="quality", goalie_pull="conservative")
    elif key == "skill_possession":
        t.update(shot_priority="quality", d_activation="active", pp_system="1-3-1")
    if pct:
        if pct.get("shooting", 50) >= 65 and key != "skill_possession":
            t["pp_system"] = "umbrella"
        if pct.get("defense", 50) >= 60 and key != "heavy_forecheck":
            t["pk_system"] = "box"
    age = _safe_float(ident.get("avg_age"), 27.0) if isinstance(ident, dict) else 27.0
    t["rest_veterans"] = age >= 28.0
    t["line_matching"] = "heavy" if _safe_float(getattr(team, "coach_rating", 72), 72) >= 78 else "light"
    if t["goalie_pull"] != "conservative":
        # Clubs chasing a playoff spot pull early.
        t["goalie_pull"] = "aggressive" if _safe_float(getattr(team, "point_pct", 0.5), 0.5) < 0.47 else "normal"
    return t


def _fit_score(pct: Dict[str, float], key: str, val: str) -> float:
    groups = _TACTIC_FIT_GROUPS.get((key, val))
    if not groups or not pct:
        return 0.0
    avg = sum(pct.get(g, 50.0) for g in groups) / len(groups)
    return _clamp((avg - 50.0) / 40.0, -1.0, 1.0)


def team_tactic_profile(team: Any) -> Dict[str, float]:
    """Per-team multipliers for its own tactics (cached on the team)."""
    t = getattr(team, "_tactics", None)
    if not isinstance(t, dict):
        t = DEFAULT_TACTICS
    ident = getattr(team, "team_identity", None)
    # Coach ratings may be added by the staff layer; read defensively.
    coach = _safe_float(getattr(team, "coach_tactics_rating", None) or getattr(team, "coach_rating", 72), 72.0)
    ckey = (tuple(str(t.get(k, "")) for k in TACTIC_OPTIONS), id(ident), round(coach, 1))
    cached = getattr(team, "_tactic_profile_cache", None)
    if isinstance(cached, tuple) and cached[0] == ckey:
        return cached[1]
    pct = _identity_pct(team)
    coach_k = _clamp(1.0 + (coach - 72.0) / 60.0, 0.75, 1.35)
    vol = xg = allow = pen = fat = 1.0
    fit_total = 0.0
    n_fit = 0
    for key, table in _TACTIC_EFFECTS.items():
        val = str(t.get(key, DEFAULT_TACTICS[key]))
        dv, dx, da, dp, df = table.get(val, (0.0, 0.0, 0.0, 0.0, 0.0))
        vol += dv
        xg += dx
        allow += da
        pen += dp
        fat += df
        if (key, val) in _TACTIC_FIT_GROUPS:
            fit_total += _fit_score(pct, key, val)
            n_fit += 1
    # Archetype fit: a system the roster suits adds execution, a mismatch costs it.
    fit = fit_total / n_fit if n_fit else 0.0
    exec_mult = 1.0 + 0.025 * fit * coach_k
    pp = 1.0 + 0.06 * _fit_score(pct, "pp_system", str(t.get("pp_system"))) * coach_k
    pk = 1.0 - 0.05 * _fit_score(pct, "pk_system", str(t.get("pk_system"))) * coach_k
    if str(t.get("pk_system")) == "pressure":
        pk *= 1.02
        pen *= 1.02
    match = _MATCH_LEVEL.get(str(t.get("line_matching")), 0.5)
    fat += 0.03 * match
    prof = {
        "vol": round(vol, 4),
        "xg": round(xg * exec_mult, 4),
        "allow": round(allow, 4),
        "pen": round(pen, 4),
        "fatigue": round(fat, 4),
        "pp": round(pp, 4),
        "pk": round(pk, 4),
        "fit": round(fit, 3),
        "match": match,
        "top_toi": round(1.0 + 0.06 * match, 4),
    }
    try:
        setattr(team, "_tactic_profile_cache", (ckey, prof))
    except Exception:
        pass
    return prof


def game_tactic_mods(home: Any, away: Any) -> Dict[str, float]:
    """Goal-rate and shot-attempt multipliers for one game (home has the last change)."""
    ph = team_tactic_profile(home)
    pa = team_tactic_profile(away)
    pp_share = 0.19  # ~0.6 of ~3.1 goals per team-game come on the PP

    def _mu(own: Dict[str, float], opp: Dict[str, float], own_lc: float, opp_lc: float) -> float:
        ev = own["vol"] * own["xg"] * opp["allow"]
        special = own["pp"] * opp["pk"] * opp["pen"]
        m = (1.0 - pp_share) * ev + pp_share * special
        # Opponent's shutdown matching (stronger with the last change) blunts your top line.
        m *= 1.0 - 0.016 * opp["match"] * opp_lc
        m *= 1.0 + 0.004 * own["match"] * own_lc
        return _clamp(m, 0.92, 1.08)

    return {
        "home_mu": _mu(ph, pa, 1.0, 0.4),
        "away_mu": _mu(pa, ph, 0.4, 1.0),
        "home_cf": _clamp(ph["vol"] * pa["allow"], 0.85, 1.15),
        "away_cf": _clamp(pa["vol"] * ph["allow"], 0.85, 1.15),
    }


def _ot_after_tie(rng: Any, hg: int, ag: int) -> Tuple[int, int, bool]:
    lo = min(hg, ag)
    if rng.random() < 0.52:
        return lo + 1, lo, True
    return lo, lo + 1, True


def apply_goalie_pull(rng: Any, hg: int, ag: int, overtime: bool, home: Any, away: Any) -> Tuple[int, int, bool]:
    """Late-game pull: shift regulation finals by the trailing coach's pull timing."""
    if overtime or hg == ag:
        return hg, ag, overtime
    trail = away if hg > ag else home
    pull = str((getattr(trail, "_tactics", None) or {}).get("goalie_pull", "normal"))
    tie_d, en_d = _PULL_DELTA.get(pull, (0.0, 0.0))
    if tie_d == 0.0 and en_d == 0.0:
        return hg, ag, overtime
    margin = abs(hg - ag)
    r = rng.random()
    if margin == 1:
        if r < tie_d:
            return _ot_after_tie(rng, hg, ag)  # extra attacker ties it
        if en_d > 0 and r < tie_d + en_d:
            return (hg + 1, ag, False) if hg > ag else (hg, ag + 1, False)  # empty-netter
    elif margin == 2:
        if en_d < 0 and r < -en_d:
            return (hg - 1, ag, False) if hg > ag else (hg, ag - 1, False)
        if en_d > 0 and r < en_d * 0.5:
            return (hg + 1, ag, False) if hg > ag else (hg, ag + 1, False)
    return hg, ag, overtime


def b2b_goalie_choice(team: Any, starter_ovr: float, backup_ovr: float) -> str:
    """'starter' or 'backup' for the second night of a back-to-back."""
    plan = str((getattr(team, "_tactics", None) or {}).get("b2b_goalie", "auto"))
    if plan in ("starter", "backup"):
        return plan
    return "starter" if (starter_ovr - backup_ovr) >= 9.0 else "backup"


def b2b_mu_mods(team: Any, *, b2b: bool, starter_ovr: float = 0.0, backup_ovr: float = 0.0) -> Tuple[float, float]:
    """(own goal-rate mult, opponent goal-rate mult) for the second night of a back-to-back."""
    if not b2b:
        return 1.0, 1.0
    own = 0.975
    choice = b2b_goalie_choice(team, starter_ovr, backup_ovr)
    setattr(team, "_tactic_b2b_goalie_choice", choice)
    if choice == "starter":
        opp = 1.03  # tired starter
    else:
        opp = 1.0 + _clamp((starter_ovr - backup_ovr) * 0.004, 0.0, 0.05)
    return own, opp
