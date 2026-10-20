"""Unit tests for draft prospect dossier OVR band integrity."""
from services.draft_prospect_profile import build_prospect_profile, _humanize_play_style


def _fixture_row(**overrides):
    base = {
        "key": "prospect-bo-zhang",
        "name": "Bo Zhang",
        "position": "LW",
        "rank": 1,
        "age": 18,
        "gp": 17,
        "goals": 15,
        "assists": 28,
        "points": 43,
        "ppg": 2.53,
        "true_ovr": 72,
        "current_ovr_estimate": 72,
        "potential_score": 86,
        "expected_ceiling_estimate": 86,
        "overall_range_low": 69,
        "overall_range_high": 75,
        "potential_range": [83, 89],
        "scouting_confidence": 62,
        "scouted_percentage": 28,
        "ceiling_hidden": False,
        "play_style": "TWO_WAY_F",
        "character_score": 55,
        "leadership": 50,
        "_prospect_revision": 42,
    }
    base.update(overrides)
    return base


def test_ovr_bands_and_headroom_delta():
    profile = build_prospect_profile(_fixture_row())

    assert profile["overallRangeLow"] == 69
    assert profile["overallRangeHigh"] == 75
    assert profile["scoutedPotentialLow"] == 83
    assert profile["scoutedPotentialHigh"] == 89
    assert profile["now_range"]["low"] == 69
    assert profile["now_range"]["high"] == 75
    assert profile["peak_range"]["high"] == 89
    assert profile["headroom_delta"] == 14
    assert profile["file_depth_label"] == "Now 69–75 OVR"
    assert profile["overallRangeLow"] <= profile["scoutedPotentialLow"]
    assert profile["overallRangeHigh"] <= profile["scoutedPotentialHigh"]


def test_analytics_scouting_history_uses_live_stats_and_revision():
    profile = build_prospect_profile(_fixture_row())
    history = profile.get("scouting_history") or []
    analytics_rows = [r for r in history if r.get("scout") == "Analytics dept."]
    assert analytics_rows, "expected analytics dept row"
    row = analytics_rows[0]
    assert "43P in 17 GP" in row["quote"]
    assert "2.53 PPG" in row["quote"]
    assert row.get("prospect_revision") == 42
    assert row.get("stats_snapshot") == {"gp": 17, "points": 43, "ppg": 2.53}


def test_play_style_humanized():
    assert _humanize_play_style("TWO_WAY_F") == "Two-way forward"
    profile = build_prospect_profile(_fixture_row())
    assert profile["play_style"] == "Two-way forward"


def test_dict_potential_range_matches_board_payload():
    profile = build_prospect_profile(
        _fixture_row(
            potential_range={"low": 75.0, "high": 90.0, "confidence": 82.0},
            potential_score=83.0,
            current_ovr_range=[76.0, 78.0],
            overall_range_low=None,
            overall_range_high=None,
        )
    )
    assert profile["scoutedPotentialLow"] == 75.0
    assert profile["scoutedPotentialHigh"] == 90.0
    assert profile["peak_range"]["high"] == 90.0
    assert profile["overallRangeLow"] == 76.0
    assert profile["overallRangeHigh"] == 78.0
    assert profile["potential"]["rating"] == 90.0


def test_character_concerns_force_sub_fifty_score_and_disastrous_read():
    profile = build_prospect_profile(
        _fixture_row(
            character_concerns=True,
            character_score=67,
            chapter_profile={"chapters": {"character": 67}},
        )
    )
    assert profile["character_read"]["character_concerns"] is True
    assert profile["character_read"]["headline"] == "Disastrous attitude"
    assert profile["character_read"]["attitude"] == "disastrous"
    char_score = profile.get("character_score")
    assert char_score is not None
    assert float(char_score) < 50
    assert any(t.get("tier") == "Disastrous" for t in profile["character_read"]["traits"])


def test_wjc_stats_block_on_profile():
    profile = build_prospect_profile(_fixture_row(
        wjc_gp=6,
        wjc_goals=4,
        wjc_assists=5,
        wjc_points=9,
        wjc_team="Canada",
        wjc_year=2025,
        wjc_result="Gold",
        wjc_stock_delta=7,
        wjc_performance_grade="breakout",
        wjc_summary="Breakout international tournament — tools and projection upgraded.",
        wjc_ovr_delta=1.2,
        wjc_potential_delta=0.8,
        wjc_ovr_before=72.0,
        wjc_ovr_after=73.2,
        wjc_potential_before=86.0,
        wjc_potential_after=86.8,
    ))
    wjc = profile.get("wjcStats") or {}
    assert wjc.get("played") is True
    assert wjc.get("games") == 6
    assert wjc.get("points") == 9
    assert wjc.get("team") == "Canada"
    assert wjc.get("ovrDelta") == 1.2
    assert wjc.get("potentialDelta") == 0.8
    assert profile.get("wjcPerformance") == wjc


# --- Scouting intel (outcomes, confidence, translation, character, leaks) ---

from services.draft_prospect_profile import (  # noqa: E402
    _outcome_distribution_block,
    _public_ovr,
    _roster_counts,
)


def test_outcomes_sum_to_100_and_name_likely_and_ceiling():
    profile = build_prospect_profile(_fixture_row())
    dist = profile["outcome_distribution"]
    assert dist["method"] == "heuristic"
    pcts = [s["pct"] for s in dist["segments"]]
    assert all(p >= 0 for p in pcts)
    assert sum(pcts) == 100
    keys = [s["key"] for s in dist["segments"]]
    assert keys == ["bust", "ahl", "mid", "top", "star"]
    assert dist["most_likely"]["key"] in keys
    assert dist["nhl_share"] == sum(s["pct"] for s in dist["segments"] if s["key"] in ("mid", "top", "star"))
    assert dist["ceiling_outcome"]["key"] in ("bust", "ahl", "mid", "top", "star")


def test_outcomes_respond_to_ceiling_and_hide_when_ceiling_hidden():
    high = _outcome_distribution_block(_fixture_row(), {"rating": 90}, 86, 92)
    low = _outcome_distribution_block(_fixture_row(), {"rating": 70}, 66, 72)
    assert high["nhl_share"] > low["nhl_share"]
    assert _outcome_distribution_block(_fixture_row(), {"hidden": True}, 80, 88) is None
    assert _outcome_distribution_block(_fixture_row(), {"rating": 80}, None, None) is None


def test_public_ovr_never_reads_unrevealed_true_ovr():
    row = _fixture_row(true_ovr=91, current_ovr_estimate=None, scouted_overall_estimate=None)
    assert _public_ovr(row) == 0.0
    assert _public_ovr({**row, "ovr_revealed": True}) == 91.0
    profile = build_prospect_profile(row)
    blob = repr(profile.get("scouting_history")) + repr(profile.get("intel"))
    assert "91" not in blob


def test_confidence_levels_follow_file_sample_and_band():
    conf = build_prospect_profile(_fixture_row(scouted_percentage=80, gp=60, points=80))["intel"]["confidence"]
    assert conf["current"]["level"] == "Confirmed"
    assert conf["translation"]["level"] in ("Reliable", "Usable")
    early = build_prospect_profile(_fixture_row(scouted_percentage=0, gp=6))["intel"]["confidence"]
    assert early["current"]["level"] == "Public only"
    assert early["translation"]["level"] == "Thin"
    hidden = build_prospect_profile(_fixture_row(ceiling_hidden=True))["intel"]["confidence"]
    assert hidden["development"]["level"] == "Unknown"


def test_character_traits_are_never_invented():
    row = _fixture_row(leadership=None)
    read = build_prospect_profile(row)["character_read"]
    assert read["modelled_count"] == 0
    assert all(not t["modelled"] and t["tier"] == "Unknown" for t in read["traits"])
    assert build_prospect_profile(row)["intel"]["confidence"]["character"]["level"] == "Not modelled"

    class _P:
        attribute_profile = {"hidden": {"coachability": 31.0, "competitiveness": 88.0}}

    read = build_prospect_profile(row, player=_P())["character_read"]
    by_label = {t["label"]: t for t in read["traits"]}
    assert by_label["Coachability"]["modelled"] and by_label["Coachability"]["score"] is None
    assert by_label["Competitive Drive"]["tier"] != "Unknown"
    assert read["modelled_count"] == 2


def test_translation_keeps_observed_and_estimates_apart():
    row = _fixture_row(gp=40, points=60, league="OHL", analytics={"war": 1.2, "xgf_pct": 55.0})
    tr = build_prospect_profile(row)["intel"]["translation"]
    assert tr["observed"]["gp"] == 40 and tr["observed"]["points"] == 60
    assert tr["estimates_label"] == "Model estimate"
    assert "war" in tr["estimates"] and "war" not in tr["observed"]
    none = build_prospect_profile(_fixture_row(gp=0, points=0))["intel"]["translation"]
    assert none["nhle"] is None and none["context_band"] is None


def test_fit_with_empty_roster_and_enum_positions():
    profile = build_prospect_profile(_fixture_row(), roster_rows=[], team_status={})
    assert "teamFit" in profile
    from enum import Enum

    class Pos(Enum):  # plain Enum: str() gives "Pos.C", like the engine's Position
        C = "C"
        D = "D"

    class Hand(Enum):
        R = "R"

    counts = _roster_counts([{"position": Pos.C}, {"position": Pos.D, "handedness": Hand.R}])
    assert counts["C"] == 1 and counts["D"] == 1 and counts["RHD"] == 1


def test_stall_scenario_moves_share_toward_current_ability():
    row = _fixture_row(gp=50)
    no_current = _outcome_distribution_block(row, {"rating": 84}, 80, 84)
    with_current = _outcome_distribution_block(row, {"rating": 84}, 80, 84, 60, 64)
    low = lambda d: sum(s["pct"] for s in d["segments"] if s["key"] in ("bust", "ahl"))
    assert low(with_current) > low(no_current)
    assert sum(s["pct"] for s in with_current["segments"]) == 100
    risky = _outcome_distribution_block({**row, "character_concerns": True}, {"rating": 84}, 80, 84, 60, 64)
    assert risky["stall_share"] > with_current["stall_share"]
