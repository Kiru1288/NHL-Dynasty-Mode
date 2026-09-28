"""Saved Edit Lines and CPU depth charts must drive per-game TOI allocation."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.sim_engine.engine import SimEngine  # noqa: E402
from services.franchise_sim import (  # noqa: E402
    _attach_franchise_saved_lineups,
    _synthetic_even_strength_lines_for_team,
)


def _fwd(pid: str, name: str, ovr: int) -> SimpleNamespace:
    ratings = {
        "skating": ovr,
        "shooting": ovr,
        "hands": ovr,
        "passing": ovr,
        "checking": ovr,
        "defense": ovr,
        "IQ": ovr,
    }
    return SimpleNamespace(
        id=pid,
        player_id=pid,
        name=name,
        position="C",
        identity=SimpleNamespace(name=name, position="C", age=24),
        ratings=ratings,
        ovr=lambda o=ovr: float(o),
        line_role="L4",
        role="depth",
    )


def _def(pid: str, name: str, ovr: int) -> SimpleNamespace:
    ratings = {
        "skating": ovr,
        "shooting": ovr - 4,
        "hands": ovr - 2,
        "passing": ovr,
        "checking": ovr,
        "defense": ovr,
        "IQ": ovr,
    }
    return SimpleNamespace(
        id=pid,
        player_id=pid,
        name=name,
        position="D",
        identity=SimpleNamespace(name=name, position="LD", age=23),
        ratings=ratings,
        ovr=lambda o=ovr: float(o),
        line_role="D3",
    )


def _goalie(pid: str, ovr: int = 82) -> SimpleNamespace:
    return SimpleNamespace(
        id=pid,
        position="G",
        identity=SimpleNamespace(name="Goalie", position="G"),
        ratings={"glove": ovr, "blocker": ovr, "rebound": ovr, "IQ": ovr},
        ovr=lambda o=ovr: float(o),
    )


def _ottawa_like_roster() -> list:
    roster = [
        _fwd("NHL_stutzle", "Stützle", 94),
        _fwd("NHL_eklund", "Eklund", 86),
        _fwd("NHL_batherson", "Batherson", 88),
        _fwd("NHL_cozens", "Cozens", 85),
        _fwd("NHL_pinto", "Pinto", 84),
        _fwd("NHL_greig", "Greig", 86),
        _fwd("NHL_burakovsky", "Burakovsky", 80),
        _fwd("NHL_amadio", "Amadio", 79),
        _fwd("NHL_zetterlund", "Zetterlund", 80),
        _fwd("NHL_coyle", "Coyle", 84),
        _fwd("NHL_foegele", "Foegele", 78),
        _fwd("NHL_halliday", "Halliday", 79),
        _def("NHL_sanderson", "Sanderson", 88),
        _def("NHL_zub", "Zub", 85),
        _def("NHL_chabot", "Chabot", 85),
        _def("NHL_spence", "Spence", 84),
        _def("NHL_kleven", "Kleven", 80),
        _def("NHL_yakemchuk", "Yakemchuk", 78),
        _goalie("NHL_ullmark", 90),
        _goalie("NHL_ersson", 84),
    ]
    # Stale tags that used to invert auto dress when saved lines failed to load.
    for p in roster:
        if "cozens" in str(p.id).lower():
            p.line_role = "L1"
            p.role = "top_line"
        if "stutzle" in str(p.id).lower():
            p.line_role = "L3"
            p.role = "middle_six"
    return roster


def _saved_user_lines() -> dict:
    return {
        "forwards": [
            {
                "id": "f1",
                "slots": {"LW": "NHL_eklund", "C": "NHL_stutzle", "RW": "NHL_batherson"},
            },
            {
                "id": "f2",
                "slots": {"LW": "NHL_burakovsky", "C": "NHL_cozens", "RW": "NHL_amadio"},
            },
            {
                "id": "f3",
                "slots": {"LW": "NHL_zetterlund", "C": "NHL_coyle", "RW": "NHL_pinto"},
            },
            {
                "id": "f4",
                "slots": {"LW": "NHL_foegele", "C": "NHL_halliday", "RW": "NHL_greig"},
            },
        ],
        "defense": [
            {"id": "d1", "slots": {"LD": "NHL_sanderson", "RD": "NHL_zub"}},
            {"id": "d2", "slots": {"LD": "NHL_chabot", "RD": "NHL_spence"}},
            {"id": "d3", "slots": {"LD": "NHL_kleven", "RD": "NHL_yakemchuk"}},
        ],
        "goalies": [{"id": "g1", "slots": {"Starter": "NHL_ullmark", "Backup": "NHL_ersson"}}],
    }


def _team_toi_for_saved_lines(sim: SimEngine, roster: list, lines: dict) -> dict:
    team = SimpleNamespace(team_id="OTT", id="OTT", roster=roster, name="Senators")
    setattr(team, "_franchise_saved_lines", lines)
    rng = random.Random(42)
    dressed, _gl, _scr, _tank = sim._gm_build_dressed_lineup(team, rng)
    sim._gm_build_game_units(dressed, team)
    toi_map = sim._gm_allocate_conserved_toi(rng, dressed, team)
    return toi_map


def test_saved_lines_l1_center_gets_more_toi_than_l2_despite_stale_roles():
    sim = SimEngine(seed=1, debug=False, populate_initial_rosters=False)
    roster = _ottawa_like_roster()
    toi = _team_toi_for_saved_lines(sim, roster, _saved_user_lines())
    st = toi.get("NHL_stutzle", 0)
    cz = toi.get("NHL_cozens", 0)
    assert st > 0 and cz > 0, (st, cz, toi)
    assert st >= cz + 120, f"expected L1C > L2C by 2+ min, got {st} vs {cz} sec"


def test_saved_lines_accept_bare_numeric_slot_ids():
    sim = SimEngine(seed=2, debug=False, populate_initial_rosters=False)
    roster = _ottawa_like_roster()
    lines = _saved_user_lines()
    lines["forwards"][0]["slots"]["C"] = "8478483"  # bare digits — roster uses NHL_stutzle style
    # Map one player to bare NHL id form
    for p in roster:
        if "stutzle" in str(p.id).lower():
            p.id = "8478483"
            p.player_id = "8478483"
    toi = _team_toi_for_saved_lines(sim, roster, lines)
    assert toi.get("8478483", 0) > toi.get("NHL_cozens", 0)


def test_cpu_synthetic_depth_star_out_tois_depth_forward():
    sim = SimEngine(seed=3, debug=False, populate_initial_rosters=False)
    roster = _ottawa_like_roster()
    synth = _synthetic_even_strength_lines_for_team(SimpleNamespace(roster=roster))
    toi = _team_toi_for_saved_lines(sim, roster, synth)
    star = toi.get("NHL_stutzle", 0)
    depth = toi.get("NHL_halliday", 0)
    assert star > depth + 180, (star, depth)


def test_cpu_synthetic_top_defenseman_out_tois_third_pair():
    sim = SimEngine(seed=4, debug=False, populate_initial_rosters=False)
    roster = _ottawa_like_roster()
    synth = _synthetic_even_strength_lines_for_team(SimpleNamespace(roster=roster))
    toi = _team_toi_for_saved_lines(sim, roster, synth)
    top = toi.get("NHL_sanderson", 0)
    third = toi.get("NHL_yakemchuk", 0)
    assert top > third + 240, (top, third)


def test_attach_lineups_sets_synthetic_for_both_teams_in_cpu_game():
    roster_h = _ottawa_like_roster()
    roster_a = [_fwd(f"NHL_a{i}", f"A{i}", 70 + i) for i in range(12)]
    roster_a += [_def(f"NHL_ad{i}", f"D{i}", 72 + i) for i in range(6)]
    roster_a += [_goalie("NHL_ag1")]
    home = SimpleNamespace(team_id="OTT", id="OTT", roster=roster_h)
    away = SimpleNamespace(team_id="TOR", id="TOR", roster=roster_a)
    session = SimpleNamespace(
        user_team_id="OTT",
        lines={
            "even_strength": {"lines": _saved_user_lines()},
        },
    )
    _attach_franchise_saved_lineups(session, home, away, home_id="OTT", away_id="TOR", user_tid="OTT")
    assert getattr(home, "_franchise_saved_lines", None) is not None
    assert getattr(away, "_franchise_saved_lines", None) is not None
    assert home._franchise_saved_lines["forwards"][0]["slots"]["C"] == "NHL_stutzle"
    away_l1_c = away._franchise_saved_lines["forwards"][0]["slots"]["C"]
    assert away_l1_c.startswith("NHL_")


def test_ld_rd_identity_does_not_steal_forward_minutes():
    """LD/RD must use pair TOI, not default-to-L3 forward minutes."""
    sim = SimEngine(seed=5, debug=False, populate_initial_rosters=False)
    roster = _ottawa_like_roster()
    toi = _team_toi_for_saved_lines(sim, roster, _saved_user_lines())
    sand = toi.get("NHL_sanderson", 0)
    zub = toi.get("NHL_zub", 0)
    chabot = toi.get("NHL_chabot", 0)
    yak = toi.get("NHL_yakemchuk", 0)
    st = toi.get("NHL_stutzle", 0)
    cz = toi.get("NHL_cozens", 0)
    pinto = toi.get("NHL_pinto", 0)
    # Zub is pair-1 RD on this sheet — he should match Sanderson, not get L3 minutes.
    assert abs(sand - zub) <= 90, (sand, zub)
    assert sand >= chabot + 90, (sand, chabot)
    assert chabot >= yak + 60, (chabot, yak)
    assert st >= cz + 120, (st, cz)
    assert cz >= pinto + 90, (cz, pinto)
    assert sand >= 22 * 60, sand
    assert st >= 19 * 60, st
    assert cz <= 19 * 60, cz
    assert pinto <= 16 * 60, pinto


def test_storyline_toi_mods_cannot_invert_saved_lines():
    sim = SimEngine(seed=6, debug=False, populate_initial_rosters=False)
    sim.set_franchise_game_stat_modifiers(
        home_player_modifiers={
            "NHL_cozens": {"toi_readiness": 0.55, "effort": 0.4, "stamina": 0.4},
            "NHL_stutzle": {"toi_readiness": -0.35, "effort": -0.25, "stamina": -0.25},
            "NHL_chabot": {"toi_readiness": 0.5, "effort": 0.4, "stamina": 0.4},
            "NHL_sanderson": {"toi_readiness": -0.3, "effort": -0.2, "stamina": -0.2},
        }
    )
    roster = _ottawa_like_roster()
    toi = _team_toi_for_saved_lines(sim, roster, _saved_user_lines())
    assert toi["NHL_stutzle"] >= toi["NHL_cozens"] + 90, (toi["NHL_stutzle"], toi["NHL_cozens"])
    assert toi["NHL_sanderson"] >= toi["NHL_spence"] - 90, (toi["NHL_sanderson"], toi["NHL_spence"])
    assert toi["NHL_sanderson"] >= toi["NHL_chabot"] + 60, (toi["NHL_sanderson"], toi["NHL_chabot"])


def test_light_path_season_toi_and_scoring_follow_lines():
    """20-game light sim: L1 TOI/scoring ahead of L2/L3; L1 cannot monopolize goals."""
    sim = SimEngine(seed=11, debug=False, populate_initial_rosters=False)
    home_roster = _ottawa_like_roster()
    away_roster = [_fwd(f"NHL_x{i}", f"X{i}", 74 + (i % 8)) for i in range(12)]
    away_roster += [_def(f"NHL_xd{i}", f"XD{i}", 73 + (i % 6)) for i in range(6)]
    away_roster += [_goalie("NHL_xg1"), _goalie("NHL_xg2")]
    home = SimpleNamespace(team_id="OTT", id="OTT", roster=home_roster, name="Ottawa")
    away = SimpleNamespace(team_id="TOR", id="TOR", roster=away_roster, name="Toronto")
    setattr(home, "_franchise_saved_lines", _saved_user_lines())
    setattr(away, "_franchise_saved_lines", _synthetic_even_strength_lines_for_team(away))
    ledger: dict = {}
    rng = random.Random(11)
    for _ in range(20):
        sim._accumulate_light_strength_game_stats(
            rng, home, away, "OTT", "TOR", hg=3, ag=2, ot=False, ledger=ledger
        )

    def _avg_toi(pid: str) -> float:
        row = ledger.get(pid) or {}
        gp = max(1, int(row.get("gp") or 0))
        return float(int(row.get("toi_sec") or 0) / gp / 60.0)

    def _pts(pid: str) -> int:
        row = ledger.get(pid) or {}
        return int(row.get("g") or 0) + int(row.get("a") or 0)

    stutzle_toi = _avg_toi("NHL_stutzle")
    cozens_toi = _avg_toi("NHL_cozens")
    pinto_toi = _avg_toi("NHL_pinto")
    halliday_toi = _avg_toi("NHL_halliday")
    sand_toi = _avg_toi("NHL_sanderson")
    yak_toi = _avg_toi("NHL_yakemchuk")
    assert stutzle_toi >= cozens_toi + 1.5, (stutzle_toi, cozens_toi)
    assert cozens_toi >= pinto_toi + 1.0, (cozens_toi, pinto_toi)
    assert pinto_toi >= halliday_toi + 0.8, (pinto_toi, halliday_toi)
    assert sand_toi >= yak_toi + 4.0, (sand_toi, yak_toi)

    l1_ids = {"NHL_stutzle", "NHL_eklund", "NHL_batherson"}
    l3_ids = {"NHL_coyle", "NHL_zetterlund", "NHL_pinto"}
    fw_ids = l1_ids | {"NHL_cozens", "NHL_burakovsky", "NHL_amadio"} | l3_ids | {
        "NHL_foegele",
        "NHL_halliday",
        "NHL_greig",
    }
    fw_goals = sum(int((ledger.get(pid) or {}).get("g") or 0) for pid in fw_ids)
    l1_goals = sum(int((ledger.get(pid) or {}).get("g") or 0) for pid in l1_ids)
    l3_goals = sum(int((ledger.get(pid) or {}).get("g") or 0) for pid in l3_ids)
    assert fw_goals >= 20, fw_goals
    assert l1_goals / fw_goals <= 0.50, (l1_goals, fw_goals)
    assert l3_goals / fw_goals >= 0.12, (l3_goals, fw_goals)
import unittest


class SavedLinesToiAllocationTests(unittest.TestCase):
    def test_saved_lines_l1_center_gets_more_toi_than_l2_despite_stale_roles(self):
        test_saved_lines_l1_center_gets_more_toi_than_l2_despite_stale_roles()

    def test_saved_lines_accept_bare_numeric_slot_ids(self):
        test_saved_lines_accept_bare_numeric_slot_ids()

    def test_cpu_synthetic_depth_star_out_tois_depth_forward(self):
        test_cpu_synthetic_depth_star_out_tois_depth_forward()

    def test_cpu_synthetic_top_defenseman_out_tois_third_pair(self):
        test_cpu_synthetic_top_defenseman_out_tois_third_pair()

    def test_attach_lineups_sets_synthetic_for_both_teams_in_cpu_game(self):
        test_attach_lineups_sets_synthetic_for_both_teams_in_cpu_game()

    def test_ld_rd_identity_does_not_steal_forward_minutes(self):
        test_ld_rd_identity_does_not_steal_forward_minutes()

    def test_storyline_toi_mods_cannot_invert_saved_lines(self):
        test_storyline_toi_mods_cannot_invert_saved_lines()

    def test_light_path_season_toi_and_scoring_follow_lines(self):
        test_light_path_season_toi_and_scoring_follow_lines()

    def test_high_potential_prospect_gets_meaningful_growth_budget_with_low_toi(self):
        test_high_potential_prospect_gets_meaningful_growth_budget_with_low_toi()


def test_high_potential_prospect_gets_meaningful_growth_budget_with_low_toi():
    from app.sim_engine.progression.development import calculate_season_growth_budget, resolve_development_profile

    prospect = SimpleNamespace(
        id="NHL_yak",
        identity=SimpleNamespace(age=21, position="D"),
        position="D",
        potential=90,
        ratings={"defense": 78, "skating": 80, "IQ": 76},
        ovr=lambda: 0.78,
        games_played=81,
        gp=81,
        toi_quality=0.32,
        role="depth",
        psych=SimpleNamespace(morale=0.62),
    )
    profile = resolve_development_profile(prospect)
    budget = calculate_season_growth_budget(
        prospect, profile=profile, rng=random.Random(9), dev_phase="NORMAL"
    )
    assert budget >= 0.035, budget


if __name__ == "__main__":
    unittest.main()
