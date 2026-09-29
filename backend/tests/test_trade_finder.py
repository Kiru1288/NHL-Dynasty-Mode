"""Trade finder: every offer it returns must be accepted by the real evaluator."""

from __future__ import annotations

import pytest

from services.franchise_sim import start_franchise
from services.trade_finder import NHL_ROSTER_MAX, _team_picks, find_trade_offers
from services.trade_service import _trade_context, build_trade_assets_payload, evaluate_franchise_trade


@pytest.fixture(scope="module")
def session():
    s = start_franchise(
        team_query="Toronto Maple Leafs",
        head_coach_name="Finder",
        coach_archetype="balanced",
        seed=42,
        warm_draft_cache=False,
    )
    build_trade_assets_payload(s)
    return s


def _assets_by_team(user_tid: str, offer: dict) -> dict:
    def payload(a):
        return {"type": a["type"], "id": a["id"], "team": a["team"]}

    return {
        offer["partner_team_id"]: [payload(a) for a in offer["user_gives"]],
        user_tid: [payload(a) for a in offer["user_gets"]],
    }


def _assert_all_accepted(session, out: dict) -> None:
    user_tid = str(session.user_team_id)
    for offer in out["offers"]:
        evaluation = evaluate_franchise_trade(session, assets_by_team=_assets_by_team(user_tid, offer))
        assert evaluation["accepted"], (offer, evaluation.get("rejection_reasons"))


def _user_players_by_ovr(session):
    user = session.team_by_id[str(session.user_team_id)]
    return sorted(user.roster, key=lambda p: -p.ovr())


def test_sell_player_offers_are_accepted(session):
    star = _user_players_by_ovr(session)[2]
    out = find_trade_offers(session, asset_type="player", asset_id=str(star.id), mode="sell")
    assert out["offers"], out.get("near_misses")
    assert len({o["partner_team_id"] for o in out["offers"]}) == len(out["offers"])
    _assert_all_accepted(session, out)


def test_sell_depth_player_finds_budget_returns(session):
    depth = _user_players_by_ovr(session)[10]
    out = find_trade_offers(session, asset_type="player", asset_id=str(depth.id), mode="sell")
    assert out["offers"]
    for offer in out["offers"]:
        assert offer["user_gets_value"] <= offer["user_gives_value"]
    _assert_all_accepted(session, out)


def test_sell_pick_offers_are_accepted(session):
    ctx = _trade_context(session)
    pick = next(r for r in _team_picks(ctx["league"], str(session.user_team_id), ctx) if int(r["round"]) == 2)
    out = find_trade_offers(session, asset_type="pick", asset_id=str(pick["pick_id"]), mode="sell")
    assert out["offers"]
    _assert_all_accepted(session, out)


def test_buy_player_respects_full_roster(session):
    user = session.team_by_id[str(session.user_team_id)]
    assert len(user.roster) >= NHL_ROSTER_MAX  # user roster is full at start
    other_tid = next(t for t in session.team_by_id if str(t) != str(session.user_team_id))
    target = sorted(session.team_by_id[other_tid].roster, key=lambda p: -p.ovr())[6]
    out = find_trade_offers(
        session, asset_type="player", asset_id=str(target.id), mode="buy", target_team_id=str(other_tid)
    )
    assert out["offers"], out.get("near_misses")
    for offer in out["offers"]:
        assert any(a["type"] == "player" and a.get("level") == "NHL" for a in offer["user_gives"])
    _assert_all_accepted(session, out)


def test_rejects_unknown_asset(session):
    with pytest.raises(ValueError):
        find_trade_offers(session, asset_type="player", asset_id="nope", mode="sell")
