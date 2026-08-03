"""Mini franchise issue audit (scoring share, aging pin, FA ceiling)."""
from __future__ import annotations

from types import SimpleNamespace

from services.contract_economy import compute_market_value
from services.franchise_sim import session_age_as_of, sync_player_age_to_season, sync_player_age_to_session


def ovr_n(o: float) -> float:
    return o / 99.0


def w_g(o: float, role: float = 1.0, hub: float = 1.0, pos: str = "C") -> float:
    on = ovr_n(o)
    shoot = on
    w = 0.52 * (on ** 1.75) + 0.28 * shoot + 0.20 * on
    w *= role * hub
    if pos == "D":
        w *= 0.30
    else:
        w *= 1.08
    if on >= 0.88:
        w *= 1.32
    elif on >= 0.84:
        w *= 1.18
    elif on >= 0.78:
        w *= 1.06
    elif on < 0.72:
        w *= 0.55
    elif on < 0.78:
        w *= 0.74
    return max(0.04, w)


def main() -> None:
    skaters = [
        (92, "C", 1.25, 1.15),
        (91, "LW", 1.2, 1.1),
        (82, "C", 1.0, 1.0),
        (80, "RW", 0.95, 1.0),
        (78, "D", 0.9, 1.0),
        (74, "C", 0.75, 1.0),
        (72, "LW", 0.7, 1.0),
        (70, "RW", 0.65, 1.0),
    ]
    weights = [w_g(o, r, h, p) for o, p, r, h in skaters]
    total = sum(weights)
    star_share = (weights[0] + weights[1]) / total
    depth_share = sum(weights[5:]) / total
    print(f"star_top2_goal_weight_share={star_share:.3f} depth_bottom3_share={depth_share:.3f}")
    assert star_share > 0.35, star_share
    assert depth_share < 0.30, depth_share

    ident = SimpleNamespace(age=23, birth_year=2002, birth_month=7, birth_day=8)
    p = SimpleNamespace(identity=ident, birth_date="2002-07-08")
    sync_player_age_to_season(p, 2026)
    sess = SimpleNamespace(
        season_calendar_year=2025,
        phase="playoff_ready",
        _year_end_progression_done=True,
        nhl_calendar=[{"iso": "2026-06-01"}],
        calendar_cursor=0,
    )
    sync_player_age_to_session(p, sess)
    print(f"age_after_playoff_serialize={ident.age} as_of={session_age_as_of(sess)}")
    assert ident.age == 24

    star = SimpleNamespace(
        id="x",
        identity=SimpleNamespace(age=28, position=SimpleNamespace(value="C")),
        ratings={"overall": 94, "dev_potential": 94},
        ovr=lambda: 0.94,
        season_stats={"pts": 110, "gp": 82},
        age=28,
        position="C",
    )
    market = compute_market_value(star)
    space = 20.0
    ceiling = min(space * 0.95, max(market * 1.18, market * 0.85 * 1.08))
    print(f"star_market={market:.2f} negotiate_ceiling={ceiling:.2f}")
    assert ceiling >= market * 0.9
    print("AUDIT_OK")


if __name__ == "__main__":
    main()
