"""Sanderson-tier D floors + Brady Tkachuk chaos house rule."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SimEngine"))

from services.brady_tkachuk_chaos import (  # noqa: E402
    BRADY_TARGET_OVR,
    apply_brady_chaos_to_league,
    brady_trade_value_override,
    display_name_with_cancer_tag,
    force_brady_overall,
    is_brady_tkachuk,
)
from services.real_nhl_roster_importer import target_ovr_from_skater_stats  # noqa: E402


def test_sanderson_tier_top_pair_d_high_80s_even_with_mid_analytics():
    """
    Jake Sanderson was ~82 because mid MoneyPuck dragged high-minute D below
    the counting/usage baseline. Top-pair minutes must floor into the high 80s.
    """
    stats = {
        "gamesPlayed": 80,
        "points": 42,
        "goals": 11,
        "assists": 31,
        "evPoints": 30,
        "ppPoints": 12,
        "pointsPerGame": 0.525,
        "timeOnIcePerGame": 1380,  # 23:00
        "hits": 70,
        "blockedShots": 120,
        "team_toi_rank_pct": 0.92,
    }
    mid_mp = {
        "games_played": 80,
        "icetime": 80 * 23 * 60,
        "gameScore": 45,
        "onIce_xGoalsPercentage": 0.49,
        "offIce_xGoalsPercentage": 0.50,
        "onIce_corsiPercentage": 0.48,
        "offIce_corsiPercentage": 0.50,
    }
    ovr, note = target_ovr_from_skater_stats(
        stats, position_code="D", age=23, analytics=mid_mp
    )
    assert ovr >= 0.875, (ovr, note)
    assert "r3" in note or "r2" in note


def test_brady_forced_55_and_cancer_name():
    class Ident:
        name = "Brady Tkachuk"

    class Fake:
        identity = Ident()
        nhl_player_id = 8480801
        ratings = {"skg_speed": 80, "off_wrist_shot_accuracy": 80, "defense": 80}
        psych = SimpleNamespace(morale=0.7, confidence=0.7, locker_room_fit=0.7)
        traits = SimpleNamespace(volatility=0.4, ego=0.4, coachability=0.6)

        def ovr(self):
            return 0.90

    # force_brady_overall needs a real Player for nudge — smoke flags/name path only
    p = Fake()
    assert is_brady_tkachuk(p)
    tagged = display_name_with_cancer_tag(p.identity.name, p)
    assert "CANCER" in tagged
    override = brady_trade_value_override(p)
    assert override is not None
    assert override["total"] < 0


def test_brady_r4_target_is_55():
    assert abs(BRADY_TARGET_OVR * 99.0 - 55.0) < 0.05


def test_brady_teammate_hit_spares_star_targets():
    from services.brady_tkachuk_chaos import degrade_teammates_for_brady
    from app.sim_engine import engine as eng

    star = SimpleNamespace(
        identity=SimpleNamespace(name="Star"),
        real_nhl_target_ovr=0.91,
        ovr=lambda: 0.91,
        ratings={"skg_speed": 90},
        in_minors=False,
        buried=False,
    )
    depth = SimpleNamespace(
        identity=SimpleNamespace(name="Depth"),
        real_nhl_target_ovr=0.72,
        ovr=lambda: 0.72,
        ratings={"skg_speed": 72},
        in_minors=False,
        buried=False,
    )
    brady = SimpleNamespace(
        identity=SimpleNamespace(name="Brady Tkachuk"),
        brady_tkachuk_chaos=True,
        nhl_player_id=8480801,
        ovr=lambda: 0.55,
        ratings={"skg_speed": 55},
        in_minors=False,
        buried=False,
    )
    team = SimpleNamespace(roster=[brady, star, depth])
    scaled = []
    real_scale = eng._scale_player_ratings

    def _scale(p, f):
        scaled.append(getattr(getattr(p, "identity", None), "name", "?"))
        return real_scale(p, f)

    eng._scale_player_ratings = _scale
    try:
        n = degrade_teammates_for_brady(team)
    finally:
        eng._scale_player_ratings = real_scale
    assert "Star" not in scaled
    assert "Depth" in scaled
    assert n >= 1
