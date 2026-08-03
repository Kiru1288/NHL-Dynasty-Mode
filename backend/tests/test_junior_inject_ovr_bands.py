"""Junior injects must use draft-eligible ages and junior-raw OVR bands."""

from __future__ import annotations

import inspect

from services import franchise_offseason as off


def test_roll_inject_source_uses_junior_raw_bands_and_draft_ages():
    src = inspect.getsource(off._roll_development_league_draft_class)
    assert "age_lo=17" in src
    assert "age_hi=18" in src
    assert "0.72, 0.86" not in src
    assert "_shape_draft_class_pipeline" in src


def test_undrafted_depth_source_uses_junior_raw_bands():
    src = inspect.getsource(off._ensure_undrafted_draft_depth)
    assert "age_lo=17" in src
    assert "0.78" not in src
    assert "_shape_draft_class_pipeline" in src


def test_eta_never_nhl_ready_for_age_16():
    from services.draft_ranking_logic import calculate_prospect_eta

    eta = calculate_prospect_eta(
        {"true_ovr": 81, "age": 16, "position": "LW", "rank": 1, "potential_score": 94}
    )
    assert eta["label"] != "Now"
    assert int(eta["years"]) >= 3
