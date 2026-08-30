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
