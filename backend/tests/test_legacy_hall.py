"""Legacy Hall / Franchise Identity rules on a small fake league (no sim needed)."""
from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace as NS

import pytest

from services import legacy_hall as L


def _player(pid, name, pos="C", lead=80):
    return NS(id=pid, name=name, position=pos, retired=False, ratings={"per_leadership": lead, "per_professionalism": 75},
              traits=None, identity=NS(name=name, age=27, position=pos), career_stats={})


def _session():
    a = NS(team_id="A", abbreviation="TOR", city="Toronto", name="Maple Leafs",
           roster=[_player("p1", "Ace One", lead=92), _player("p2", "Bo Two", lead=70), _player("g1", "Gee", "G", 60)])
    b = NS(team_id="B", abbreviation="MTL", city="Montreal", name="Canadiens",
           roster=[_player("p3", "Cy Three", lead=85), _player("p4", "Di Four", lead=65)])
    league = NS(teams=[a, b], trade_history=[])
    s = NS(sim=NS(league=league), team_by_id={"A": a, "B": b}, user_team_id="A", season_calendar_year=2026,
           season_history=[], nhl_calendar=[{"iso": "2027-04-20"}], calendar_cursor=0, retired_players_archive=[],
           awards_payload={}, draft_state={}, game_results=[], champion_id="A", playoff_live=None,
           playoff_payload={"series": [{"round_index": 1, "team_high_id": "A", "team_low_id": "B", "wins_high": 4, "wins_low": 3, "winner_id": "A"}]})
    s.standings = NS(records={"A": NS(wins=50, losses=25, otl=7, points=107, gf=280, ga=230),
                              "B": NS(wins=40, losses=35, otl=7, points=87, gf=240, ga=250)})
    s.player_season_stats = {
        "p1": {"gp": 82, "g": 40, "a": 50, "pts": 90, "team_id": "A", "name": "Ace One", "position": "C"},
        "p3": {"gp": 80, "g": 30, "a": 30, "pts": 60, "team_id": "B", "name": "Cy Three", "position": "C"},
        "g1": {"gp": 60, "w": 35, "l": 20, "so": 4, "ga": 150, "shots_against": 1700, "team_id": "A", "name": "Gee", "position": "G"},
    }
    s.game_results = [{"home_id": "A", "away_id": "B", "home_score": 3, "away_score": 2, "stat_scope": "regular_season"}] * 4
    return s


def test_finalize_is_idempotent_and_records_champion():
    s = _session()
    row = L.archive_legacy_season(s, 2026)
    snap = copy.deepcopy(s.legacy)
    L.archive_legacy_season(s, 2026)
    assert s.legacy["seasons"] == snap["seasons"]
    assert row["champion_id"] == "A" and row["finalist_id"] == "B"
    assert sum(1 for e in s.legacy["events"].values() if e["type"] == "championship") == 1
    assert s.legacy["players"]["p1"]["championships"] == [2026]


def test_regular_season_only_and_missing_is_not_zero():
    s = _session()
    L.archive_legacy_season(s, 2026)
    d = L.player_dossier(s, "p1")
    assert d["career_in_save"]["pts"] == 90 and d["coverage"]["playoffs"] is False
    assert all(x["scope"] == "regular" for x in d["seasons"])
    # a team never seen in a finalized season has no rows, not zero rows of zeros
    assert L.franchise_summary(s, "A")["seasons"][0]["pts"] == 107


def test_rebrand_keeps_id_history_and_is_atomic():
    s = _session()
    L.archive_legacy_season(s, 2026)
    s.season_calendar_year = 2027
    before_team = (s.team_by_id["A"].name, s.team_by_id["A"].abbreviation, [p.id for p in s.team_by_id["A"].roster])
    ident = copy.deepcopy(L.current_identity(s, "A"))
    bad = dict(ident, abbreviation="nope99")
    assert L.rebrand(s, "A", bad)["ok"] is False
    assert s.legacy["franchises"]["A"]["current_version"] == 1  # nothing partially applied
    ident["name"], ident["abbreviation"] = "Toronto Lynx", "TLX"
    out = L.rebrand(s, "A", ident, request_id="r1")
    again = L.rebrand(s, "A", ident, request_id="r1")
    assert out["ok"] and out["version"] == 2 and again.get("duplicate")
    assert (s.team_by_id["A"].name, s.team_by_id["A"].abbreviation, [p.id for p in s.team_by_id["A"].roster]) == before_team
    # the 2026 season keeps the identity it was played under
    sv = L.season_view(s, 2026)
    assert sv["team_names"]["A"] == "Toronto Maple Leafs"
    assert L.current_identity(s, "A")["name"] == "Toronto Lynx"
    assert sum(1 for e in s.legacy["events"].values() if e["type"] == "rebrand") == 1


def test_contrast_and_input_validation():
    s = _session()
    ident = copy.deepcopy(L.current_identity(s, "A"))
    ident["colors"]["secondary"] = ident["colors"]["primary"]
    _, errs = L.validate_identity(ident)
    assert any("contrast" in e for e in errs)
    ident = copy.deepcopy(L.current_identity(s, "A"))
    ident["crest"]["layers"] = [{"kind": "evil"}]
    _, errs = L.validate_identity(ident)
    assert any("kind must be" in e for e in errs)


def test_captaincy_history_survives_change_and_departure():
    s = _session()
    L.ensure_legacy(s)
    L.appoint_captaincy(s, "A", "p1", ["p2"])
    L.appoint_captaincy(s, "A", "p2", [])
    hist = L.current_captains(s, "A")["history"]
    assert [t["player_id"] for t in hist] == ["p1", "p2"] and hist[0]["end_reason"] == "Replaced"
    assert s.team_by_id["A"].roster[1].is_captain is True and s.team_by_id["A"].roster[0].is_captain is False
    s.team_by_id["A"].roster = [p for p in s.team_by_id["A"].roster if p.id != "p2"]  # traded away
    L.reconcile_captaincy(s)
    assert L.current_captains(s, "A")["captain"] is None  # user team is never auto-appointed
    assert L.current_captains(s, "A")["history"][-1]["end_reason"] in ("Left the club", "Retired")
    with pytest.raises(ValueError):
        L.appoint_captaincy(s, "A", "p3", [])  # not on this roster


def test_cpu_clubs_get_evidence_based_captain():
    s = _session()
    L.reconcile_captaincy(s, appoint_cpu=True)
    assert L.current_captains(s, "B")["captain"]["player_id"] == "p3"  # highest leadership, not OVR
    assert L.current_captains(s, "A")["captain"] is None


def test_backfill_old_save_flags_partial():
    s = _session()
    s.season_history = [{"season_year": 2025, "champion_id": "B", "teams": {"A": {"w": 40, "l": 30, "otl": 12, "pts": 92}}}]
    lg = L.ensure_legacy(s)
    assert lg["seasons"]["2025"]["source"] == "backfill" and lg["seasons"]["2025"]["coverage"]["partial"]
    assert lg["seasons"]["2025"]["standings"]["A"]["made_playoffs"] is None  # unknown, not False
    assert 2025 in lg["coverage"]["backfilled_seasons"]


def test_trivia_is_sourced_and_deterministic():
    s = _session()
    L.archive_legacy_season(s, 2026)
    t1 = L.trivia(s, "A")
    t2 = L.trivia(s, "A")
    assert t1 == t2 and t1["facts"]
    assert all(f["source"] and f["id"].startswith("tv_A_") for f in t1["facts"])


def test_rivalry_needs_evidence_and_is_bounded():
    s = _session()
    L.archive_legacy_season(s, 2026)
    r = L.rivalries(s, "A")["rivalries"]
    assert r and 0 <= r[0]["score"] <= 100 and r[0]["inputs"]["series"] == 1 and r[0]["inputs"]["seven_game"] == 1


def test_milestone_dedup_and_pickle_roundtrip():
    s = _session()
    s.player_season_stats["p1"]["pts"] = 520
    L.archive_legacy_season(s, 2026)
    n = sum(1 for e in s.legacy["events"].values() if e["type"] == "career_milestone")
    L._season_milestones(s, s.legacy, 2026)
    assert n == sum(1 for e in s.legacy["events"].values() if e["type"] == "career_milestone") and n >= 1
    clone = pickle.loads(pickle.dumps(s.legacy))
    assert clone["seasons"]["2026"]["champion_id"] == "A"


def test_lean_block_is_small():
    s = _session()
    L.ensure_legacy(s)
    import json
    assert len(json.dumps(L.lean_identity_block(s))) < 600


def test_event_headlines_keep_the_name_of_their_time():
    s = _session()
    L.archive_legacy_season(s, 2026)
    s.season_calendar_year = 2027
    ident = copy.deepcopy(L.current_identity(s, "A"))
    ident["name"], ident["abbreviation"] = "Toronto Lynx", "TLX"
    assert L.rebrand(s, "A", ident)["ok"]
    heads = [e["headline"] for e in L.timeline(s, team_id="A")["events"]]
    assert any(h.startswith("Toronto Maple Leafs win the 2026-27") for h in heads)
    assert any(h.startswith("Toronto Lynx unveil") for h in heads)
