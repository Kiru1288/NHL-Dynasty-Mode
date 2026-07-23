"""Draft class ranking audit — generation sanity checks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "SimEngine"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

from services.draft_ranking_logic import (
    apply_hard_ranking_floor_pass,
    apply_ranking_sanity_pass,
    build_potential_intel,
    calculate_prospect_eta,
    clean_team_name,
    compose_live_draft_board,
    compute_enhanced_draft_score,
    log_draft_class_audit,
    normalize_league_code,
    size_score_modifier,
)


def _true_pot(row):
    return float(row.get("true_potential_score") or row.get("potential_score") or 0)


def test_clean_team_strips_league_junk():
    parts = normalize_league_code("EU_J_SK", "Slovakia U20")
    assert "Pro Jr" not in parts["league_display"]
    assert parts["league_display"] == "Slovak Extraliga"
    team = clean_team_name("Lethbridge EU J SK 2", "EU_J_SK", parts["league_display"])
    assert "EU" not in team.upper().split()
    assert team.startswith("Lethbridge")


def test_normalize_league_code_no_hybrid_labels():
    for code, expected in (
        ("EU_J_SHL", "J20 Nationell"),
        ("EU_J_LIIGA", "U20 SM-sarja"),
        ("EU_J_KHL_JR", "MHL"),
        ("CHL_OHL", "OHL"),
    ):
        parts = normalize_league_code(code, "")
        assert "Pro Jr" not in parts["league_display"]
        assert parts["league_display"] == expected


def test_prospect_pipeline_audit_500():
    import random

    from app.sim_engine.generation.prospect_league_teams import (
        LEAGUE_REGISTRY,
        apply_prospect_league_team_fix,
        choose_nationality_for_league,
        league_fit_tier,
        teams_for_league,
        validate_prospect_league_fit,
    )
    from app.sim_engine.generation.prospect_league_scoring import _compute_stock_fields

    rng = random.Random(20260701)
    prospects = []
    per_league = max(1, 520 // max(1, len(LEAGUE_REGISTRY)))

    for code in LEAGUE_REGISTRY:
        for _ in range(per_league):
            nat = choose_nationality_for_league(rng, code)
            teams = teams_for_league(code)
            team = rng.choice(teams) if teams else {"name": "Unknown"}
            row = {
                "key": f"{code}-{len(prospects)}",
                "nationality": nat,
                "country": nat,
                "league_code": code,
                "team": team["name"],
                "team_name": team["name"],
            }
            if not validate_prospect_league_fit(nat, code):
                row = apply_prospect_league_team_fix(row)
            else:
                row = apply_prospect_league_team_fix(row)
            prospects.append(row)

    assert len(prospects) >= 500
    invalid = sum(1 for p in prospects if p.get("league_fit_tier") == "invalid")
    rare = sum(1 for p in prospects if p.get("league_fit_tier") == "rare_import")
    assert invalid == 0
    assert rare < len(prospects) * 0.12

    by_nat: dict = {}
    for p in prospects:
        nat = str(p.get("nationality") or p.get("country") or "Unknown")
        code = str(p.get("league_code") or "")
        by_nat.setdefault(nat, []).append(code)

    can_chl = sum(
        1 for code in by_nat.get("Canada", [])
        if code.startswith("CHL_") or code in ("USHL", "NCAA")
    )
    assert can_chl >= len(by_nat.get("Canada", [])) * 0.90

    usa_na = sum(
        1 for code in by_nat.get("USA", [])
        if code in ("USHL", "NCAA") or code.startswith("CHL_")
    )
    assert usa_na >= len(by_nat.get("USA", [])) * 0.85

    swe_dom = sum(1 for code in by_nat.get("Sweden", []) if code.startswith("EU_J_"))
    assert swe_dom >= len(by_nat.get("Sweden", [])) * 0.70

    rus_mhl = sum(1 for code in by_nat.get("Russia", []) if code == "EU_J_KHL_JR")
    assert rus_mhl >= len(by_nat.get("Russia", [])) * 0.55


def test_stock_movement_spread_is_dramatic():
    from types import SimpleNamespace

    from app.sim_engine.generation.prospect_league_scoring import _compute_stock_fields

    class _Prospect:
        position = "C"
        character_concerns = False
        pipeline_bust = False

    deltas = []
    for i in range(120):
        gp = 8 + (i % 12)
        mode = i % 3
        if mode == 0:
            actual = {
                "gp": gp,
                "points": 2 + (i % 3),
                "goals": 1,
                "assists": 1,
                "ppg": (2 + (i % 3)) / gp,
                "cf_pct": 41 + (i % 4),
                "xgf_pct": 40 + (i % 5),
                "primary_points": 2,
            }
            projected = {"gp": 60, "points": 48, "ppg": 0.80, "goals": 22, "assists": 26}
        elif mode == 1:
            actual = {
                "gp": gp,
                "points": 6 + (i % 8),
                "goals": 2,
                "assists": 4 + (i % 6),
                "ppg": (6 + (i % 8)) / gp,
                "cf_pct": 56 + (i % 6),
                "xgf_pct": 55 + (i % 7),
                "primary_points": 5 + (i % 6),
            }
            projected = {"gp": 60, "points": 34, "ppg": 0.57, "goals": 14, "assists": 20}
        else:
            actual = {
                "gp": gp,
                "points": 18 + (i % 16),
                "goals": 8 + (i % 8),
                "assists": 10 + (i % 10),
                "ppg": (18 + (i % 16)) / gp,
                "cf_pct": 52 + (i % 10),
                "xgf_pct": 53 + (i % 11),
                "primary_points": 14 + (i % 12),
            }
            projected = {"gp": 60, "points": 36, "ppg": 0.60, "goals": 16, "assists": 20}
        out = _compute_stock_fields(_Prospect(), actual, projected, "CHL_OHL")
        deltas.append(int(out.get("stock_delta") or 0))

    assert max(deltas) >= 12
    assert min(deltas) <= -8
    holding = sum(1 for d in deltas if -2 <= d <= 2)
    assert holding < len(deltas) * 0.35
    assert any(d >= 20 for d in deltas) or any(d <= -18 for d in deltas)


def test_low_potential_penalized_in_sanity_pass():
    rows = [
        {"key": "a", "name": "High", "true_ovr": 74, "true_potential_score": 84, "potential_score": 84, "production_adjusted_score": 1.1, "position": "C", "age": 18},
        {"key": "b", "name": "Low", "true_ovr": 70, "true_potential_score": 52, "potential_score": 52, "production_adjusted_score": 0.5, "position": "RW", "age": 18},
    ]
    for r in rows:
        r["_score"] = compute_enhanced_draft_score(r)
    rows.sort(key=lambda x: -x["_score"])
    apply_ranking_sanity_pass(rows)
    assert rows[0]["key"] == "a"
    low_rank = next(i for i, r in enumerate(rows) if r["key"] == "b") + 1
    assert low_rank > 1


def test_short_forward_without_production_gets_penalty():
    row = {"position": "RW", "height_cm": 173, "production_adjusted_score": 0.6, "ppg": 0.5}
    assert size_score_modifier(row) < -3


def test_intel_range_widens_at_low_confidence():
    intel = build_potential_intel(82.0, 40.0)
    assert intel["potential_range"]["high"] - intel["potential_range"]["low"] > 8
    assert intel["intel_label"] == "Limited"


def test_hard_floor_demotes_low_potential_top20():
    rows = []
    for i in range(25):
        rows.append({
            "key": f"good{i}",
            "name": f"Good{i}",
            "position": "C",
            "true_ovr": 72,
            "true_potential_score": 82,
            "potential_score": 82,
            "production_adjusted_score": 1.0,
            "_score": 80 - i * 0.1,
        })
    for i in range(5):
        rows.append({
            "key": f"bad{i}",
            "name": f"Bad{i}",
            "position": "C",
            "true_ovr": 50,
            "true_potential_score": 52,
            "potential_score": 52,
            "production_adjusted_score": 2.0,
            "_score": 95 - i,
        })
    apply_hard_ranking_floor_pass(rows)
    top10_pot = [_true_pot(r) for r in rows[:10]]
    assert min(top10_pot) >= 70


def test_calculate_prospect_eta_top10_skater():
    row = {"position": "C", "true_ovr": 70, "true_potential_score": 82, "age": 18, "league_code": "CHL_OHL"}
    eta = calculate_prospect_eta(row, final_rank=5)
    assert int(eta["years"]) <= 2


def test_compose_board_includes_minimum_goalies():
    rows = [{"key": f"s{i}", "position": "C", "_score": 200 - i} for i in range(350)]
    rows += [{"key": f"g{i}", "position": "G", "_score": 30 - i} for i in range(20)]
    board = compose_live_draft_board(rows, min_goalies=12, target_goalies=16)
    g_count = sum(1 for r in board if r.get("position") == "G")
    assert g_count >= 12
    top_g_rank = min(i + 1 for i, r in enumerate(board) if r.get("position") == "G")
    assert top_g_rank >= 96


def test_hard_floor_demotes_low_potential_top32():
    rows = []
    for i in range(40):
        rows.append({
            "key": f"good{i}",
            "name": f"Good{i}",
            "position": "C",
            "true_ovr": 72,
            "true_potential_score": 78,
            "potential_score": 78,
            "production_adjusted_score": 1.0,
            "_score": 80 - i * 0.1,
        })
    rows.append({
        "key": "bad",
        "name": "Bad",
        "position": "C",
        "true_ovr": 62,
        "true_potential_score": 49,
        "potential_score": 49,
        "production_adjusted_score": 1.4,
        "_score": 95,
    })
    apply_hard_ranking_floor_pass(rows)
    top32_pot = [_true_pot(r) for r in rows[:32]]
    assert min(top32_pot) >= 60


def test_audit_logs_goalie_count():
    rows = [{"position": "G", "potential_score": 80, "true_potential_score": 80, "name": "G1"}] * 14
    rows += [{"position": "C", "potential_score": 75, "true_potential_score": 75, "name": f"F{i}"} for i in range(20)]
    audit = log_draft_class_audit(rows)
    assert audit["goalie_count"] == 14
