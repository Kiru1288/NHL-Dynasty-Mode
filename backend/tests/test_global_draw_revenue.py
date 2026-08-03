"""Yao Ming / global-draw revenue + signing-bonus revenue tiers."""

from __future__ import annotations

from types import SimpleNamespace

from services.franchise_offseason import (
    SIGNING_BONUS_REVENUE_FLOOR_M,
    signing_bonus_max_pct_for_revenue,
)
from services.league_operations import calculate_global_draw_revenue_boost


def _player(*, name: str, country: str, ovr: float):
    return SimpleNamespace(
        name=name,
        overall=ovr,
        retired=False,
        identity=SimpleNamespace(birth_country=country),
    )


def test_global_draw_boost_for_chinese_star():
    team = SimpleNamespace(
        roster=[
            _player(name="Local Star", country="Canada", ovr=92),
            _player(name="Yao Effect", country="China", ovr=88),
        ]
    )
    result = calculate_global_draw_revenue_boost(team)
    assert result["global_draw_revenue_boost"] >= 20.0
    assert "Global Draw" in result["global_draw_tags"]
    assert any(p["name"] == "Yao Effect" for p in result["global_draw_players"])


def test_no_global_draw_for_standard_nations():
    team = SimpleNamespace(
        roster=[
            _player(name="Swede", country="Sweden", ovr=90),
            _player(name="Canadian", country="Canada", ovr=91),
        ]
    )
    result = calculate_global_draw_revenue_boost(team)
    assert result["global_draw_revenue_boost"] == 0.0
    assert result["global_draw_players"] == []


def test_signing_bonus_tiers_scale_with_revenue():
    assert signing_bonus_max_pct_for_revenue(SIGNING_BONUS_REVENUE_FLOOR_M - 1) == 0.0
    assert signing_bonus_max_pct_for_revenue(160) == 0.08
    assert signing_bonus_max_pct_for_revenue(175) == 0.14
    assert signing_bonus_max_pct_for_revenue(195) == 0.20
    assert signing_bonus_max_pct_for_revenue(215) == 0.26
    assert signing_bonus_max_pct_for_revenue(240) == 0.32


if __name__ == "__main__":
    test_global_draw_boost_for_chinese_star()
    test_no_global_draw_for_standard_nations()
    test_signing_bonus_tiers_scale_with_revenue()
    print("OK")
