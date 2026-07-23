"""
Shared draft board engine — team-specific scouting uncertainty.

Public consensus ≠ internal board. Each club gets noisy attribute estimates that
drive reaches, steals, and philosophy differences.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def _rng(seed_parts: Any) -> float:
    raw = ":".join(str(p) for p in seed_parts) if isinstance(seed_parts, (list, tuple)) else str(seed_parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def team_scouting_estimate(
    *,
    team_id: str,
    prospect_id: str,
    true_ovr: float,
    true_potential: float,
    public_rank: int,
    scouting_quality: float = 60.0,
    draft_year: int = 2026,
) -> Dict[str, Any]:
    """
    Build a team-private estimate of a prospect. Higher scouting_quality → tighter noise.
    """
    q = max(20.0, min(95.0, float(scouting_quality)))
    noise_amp = (100.0 - q) / 100.0
    n1 = _rng((team_id, prospect_id, draft_year, "ovr")) - 0.5
    n2 = _rng((team_id, prospect_id, draft_year, "pot")) - 0.5
    n3 = _rng((team_id, prospect_id, draft_year, "risk"))

    ovr_err = n1 * 18.0 * noise_amp
    pot_err = n2 * 22.0 * noise_amp
    scouted_ovr = max(40.0, min(95.0, float(true_ovr) + ovr_err))
    pot_lo = max(45.0, min(99.0, float(true_potential) + pot_err - 4.0 * noise_amp))
    pot_hi = max(pot_lo + 1.0, min(99.0, float(true_potential) + pot_err + 6.0 * noise_amp))
    confidence = max(0.15, min(0.98, q / 100.0 - abs(n1) * 0.25 * noise_amp))

    risk = "Low"
    if n3 > 0.72 or (true_potential - true_ovr) > 18:
        risk = "High"
    elif n3 > 0.45:
        risk = "Medium"

    # Rank perception drifts from public list based on quality.
    rank_shift = int(round((n1 + n2) * 12.0 * noise_amp))
    team_board_bias_rank = max(1, int(public_rank) + rank_shift)

    return {
        "scouted_overall_estimate": round(scouted_ovr, 1),
        "scouted_potential_range": [round(pot_lo, 1), round(pot_hi, 1)],
        "scouting_confidence": round(confidence * 100.0, 1),
        "risk_assessment": risk,
        "team_board_rank_hint": team_board_bias_rank,
        "public_rank": int(public_rank),
    }


def enrich_board_entry_with_team_scouting(
    entry: Dict[str, Any],
    *,
    team_id: str,
    scouting_quality: float = 60.0,
    draft_year: int = 2026,
    interview: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pid = str(entry.get("key") or entry.get("prospect_id") or entry.get("id") or "")
    true_ovr = float(entry.get("true_ovr") or entry.get("nhl_readiness") or 55)
    true_pot = float(entry.get("potential_score") or entry.get("ceiling") or true_ovr + 8)
    pub = int(entry.get("rank") or entry.get("public_rank") or 99)
    est = team_scouting_estimate(
        team_id=team_id,
        prospect_id=pid,
        true_ovr=true_ovr,
        true_potential=true_pot,
        public_rank=pub,
        scouting_quality=scouting_quality,
        draft_year=draft_year,
    )
    out = {**entry, **est}
    if interview:
        out["interview"] = interview
        # Interview quality nudges internal confidence and board hint
        quality = str(interview.get("quality") or interview.get("impression") or "").lower()
        if "strong" in quality or "positive" in quality:
            out["scouting_confidence"] = min(98.0, float(out["scouting_confidence"]) + 6.0)
            out["team_board_rank_hint"] = max(1, int(out["team_board_rank_hint"]) - 2)
            out["willingness_to_sign"] = interview.get("willingness_to_sign", out.get("willingness_to_sign"))
        elif "concern" in quality or "poor" in quality or "flag" in quality:
            out["scouting_confidence"] = max(12.0, float(out["scouting_confidence"]) - 8.0)
            out["team_board_rank_hint"] = int(out["team_board_rank_hint"]) + 3
            out["character_concerns"] = True
        if interview.get("medical_flag"):
            out["medical_concerns"] = True
            out["scouting_confidence"] = max(10.0, float(out["scouting_confidence"]) - 5.0)
        if interview.get("ncaa_commitment"):
            out["ncaa_commitment"] = True
        if interview.get("european_contract"):
            out["european_contract"] = True
        if interview.get("market_preference"):
            out["market_preference"] = interview.get("market_preference")
    return out


def score_candidate_for_team(
    entry: Dict[str, Any],
    *,
    overall_pick: int,
    philosophy: str = "bpa_heavy",
    needs: Optional[List[Dict[str, Any]]] = None,
    team_board_score: float = 0.0,
    rng_noise: float = 0.0,
) -> float:
    """Shared CPU scoring used by franchise mode and universe draft adapters."""
    needs = needs or []
    pub_rank = int(entry.get("rank") or entry.get("public_rank") or 999)
    scout_rank = int(entry.get("team_board_rank") or entry.get("team_board_rank_hint") or pub_rank)
    conf = float(entry.get("scouting_confidence") or 50) / 100.0

    bpa = max(0.0, 120.0 - scout_rank * 1.15)
    if overall_pick <= 10:
        bpa *= 1.35
    elif overall_pick <= 32:
        bpa *= 1.1

    upside_raw = entry.get("scouted_potential_range")
    if isinstance(upside_raw, (list, tuple)) and len(upside_raw) >= 2:
        upside_v = float(upside_raw[1]) * 0.4
    elif isinstance(upside_raw, (list, tuple)) and len(upside_raw) >= 1:
        upside_v = float(upside_raw[0]) * 0.4
    else:
        upside_v = float(entry.get("potential_score") or 0) * 0.4
    # Prefer the anti-correlated floor estimate when available (a reliable projectable
    # outcome), falling back to current ability. This is what makes a high-floor prospect a
    # legitimate alternative to a boom/bust swing.
    floor_score = float(entry.get("floor_score") or entry.get("scouted_overall_estimate") or entry.get("true_ovr") or 0)
    floor_v = floor_score * 0.35
    # Later picks increasingly value a high, reliable floor over raw ceiling — a safe
    # contributor is worth more than a long-shot swing once the elite talent is gone.
    late_factor = max(0.0, min(1.0, (overall_pick - 20) / 60.0))
    floor_incentive = floor_score * (0.10 + 0.45 * late_factor)
    # Symmetrically, discount pure ceiling on late picks so upside doesn't always win.
    upside_v *= (1.0 - 0.35 * late_factor)

    risk_pen = 4.0 if str(entry.get("risk_assessment") or entry.get("risk") or "") == "High" else 0.0
    if entry.get("character_concerns") or entry.get("medical_concerns"):
        risk_pen += 5.0
    if entry.get("willingness_to_sign") is False:
        risk_pen += 4.0

    need_boost = 0.0
    pos = str(entry.get("position") or "").upper()
    for n in needs[:3]:
        cat = n.get("category") if isinstance(n, dict) else str(n)
        pri = float(n.get("priority") or 1.0) if isinstance(n, dict) else 1.0
        if cat in ("Franchise Center", "Center Depth") and pos == "C":
            need_boost += 8.0 * pri
        if cat == "Right-Shot Defense" and pos == "D":
            need_boost += 7.0 * pri
        if cat == "Goalie Pipeline" and pos == "G":
            need_boost += 5.0 * pri
        if cat in ("Top-Six Winger", "Wing Depth") and pos in ("LW", "RW", "W"):
            need_boost += 5.0 * pri
        if cat in ("Near-Ready Help", "Young NHL Depth") and float(entry.get("true_ovr") or entry.get("scouted_overall_estimate") or 0) >= 70:
            need_boost += 3.0 * pri
        if cat == "High-Upside Swing":
            need_boost += upside_v * 0.05 * pri

    # Goalies compete on board score + need. Only mild early-pick caution for weak goalies.
    goalie_pen = 0.0
    if pos == "G" and overall_pick <= 15 and not entry.get("generational_goalie"):
        pot = 0.0
        upside_raw = entry.get("scouted_potential_range")
        if isinstance(upside_raw, (list, tuple)) and upside_raw:
            pot = float(upside_raw[-1] if len(upside_raw) >= 2 else upside_raw[0])
        else:
            pot = float(entry.get("potential_score") or 0)
        floor_g = float(entry.get("scouted_overall_estimate") or entry.get("true_ovr") or 0)
        if pot < 80 and floor_g < 70:
            goalie_pen = 3.0 + overall_pick * 0.05
        # Strong goalies: no artificial demotion — need_boost / BPA score decide.

    phil = str(philosophy or "")
    phil_boost = 0.0
    band = str(entry.get("outcome_band") or "")
    if phil == "safe_floor" and (str(entry.get("risk_assessment") or entry.get("risk") or "") == "Low" or band == "Safe Floor"):
        phil_boost += 4.0 + floor_score * 0.06
    if phil in ("high_upside", "boom_bust_gambler", "rebuilder_upside"):
        phil_boost += upside_v * 0.08
    if phil == "contender_timeline" and conf > 0.75:
        phil_boost += 3.0
    if phil == "center_priority" and pos == "C":
        phil_boost += 5.0
    if phil == "defense_first" and pos == "D":
        phil_boost += 4.0

    chaos = rng_noise * (14.0 if phil == "boom_bust_gambler" else 6.0)

    return (
        bpa
        + float(team_board_score) * 0.55
        + upside_v * 0.25
        + floor_v * 0.2
        + floor_incentive
        + need_boost
        + conf * 4.0
        + phil_boost
        - risk_pen
        - goalie_pen
        + chaos
    )


def select_best_prospect(
    available: List[Dict[str, Any]],
    *,
    overall_pick: int,
    philosophy: str,
    needs: Optional[List[Dict[str, Any]]] = None,
    noise_fn=None,
) -> Dict[str, Any]:
    if not available:
        raise ValueError("No prospects available")
    scored = []
    for e in available:
        noise = 0.0
        if callable(noise_fn):
            noise = float(noise_fn(e) or 0.0)
        scored.append(
            (
                e,
                score_candidate_for_team(
                    e,
                    overall_pick=overall_pick,
                    philosophy=philosophy,
                    needs=needs,
                    team_board_score=float(e.get("team_board_score") or 0),
                    rng_noise=noise,
                ),
            )
        )
    scored.sort(key=lambda x: -x[1])
    return scored[0][0]
