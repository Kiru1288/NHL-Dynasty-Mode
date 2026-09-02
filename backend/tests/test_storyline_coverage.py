"""Unit tests for published storyline coverage: traits, boxes, insiders payload."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "1")

from app.sim_engine.franchise.storyline_coverage import (  # noqa: E402
    coverage_payload_fields,
    ingest_game_box_storylines,
    personality_from_player,
)
from app.sim_engine.franchise.storyline_engine import _u_personality  # noqa: E402


class _Traits:
    loyalty = 0.9
    ego = 0.2
    volatility = 0.15
    competitiveness = 0.8
    leadership = 0.7
    ambition = 0.85
    coachability = 0.75
    media_comfort = 0.6
    introversion = 0.2
    family_priority = 0.7
    work_ethic = 0.8
    mental_toughness = 0.65
    money_focus = 0.3
    clutch_tendency = 0.55
    patience = 0.6


class _Psych:
    morale = 0.7
    role_satisfaction = 0.3
    ice_time_satisfaction = 0.25
    coach_trust = 0.4
    trust_in_management = 0.35
    trust_in_teammates = 0.8
    locker_room_fit = 0.2
    media_stress = 0.4
    confidence_level = 0.55
    contract_pressure = 0.8


class _Ident:
    name = "Alex Example"
    age = 26
    birth_city = "Winnipeg"
    birth_country = "Canada"
    draft_year = 2018
    draft_round = 1
    draft_pick = 8


class _Player:
    id = "p_test_1"
    name = "Alex Example"
    traits = _Traits()
    psych = _Psych()
    identity = _Ident()
    position = "C"
    overall = 82


class _Session:
    def __init__(self):
        self.user_team_id = "T1"
        self.season_calendar_year = 2026
        self.calendar_cursor = 20
        self.nhl_calendar = [{"iso": "2026-10-20"}] * 40
        self.storyline_events = []
        self.knowledge_graph = []
        self.universe_players = {
            "p_test_1": {
                "player_id": "p_test_1",
                "player_name": "Alex Example",
                "team_id": "T1",
                "active_roster": True,
                "overall": 82,
                "position": "C",
                "identity": {"name": "Alex Example", "birth_city": "Winnipeg"},
                "trusts": {"coach": 40, "gm": 35, "teammates": 80, "room": 20},
                "top_concerns": [{"id": "role", "label": "Role and ice time", "pressure": 62}],
                "memories": [{"summary": "Sat a healthy scratch"}],
                "personality_tags": ["Demanding"],
                "reputation_tags": ["Demanding"],
                "niche_abilities": [{"label": "Spark Plug"}],
            }
        }
        self._storyline_fired = {}
        self.player_recent_games = {}
        self.player_season_stats = {}
        self.team_by_id = {}
        self.decision_event_log = []
        self.social_posts = []


def test_personality_reads_real_traits_not_hash():
    player = _Player()
    first = personality_from_player(player)
    second = _u_personality(player, "p_test_1")
    assert first["loyalty"] == second["loyalty"] == 90.0
    assert first["ego"] == 20.0
    assert first["sociability"] == 80.0
    assert first["ambition"] == 85.0
    hashed = _u_personality(player, "totally-different-id")
    assert hashed["loyalty"] == first["loyalty"]


def test_hat_trick_from_scoring_events():
    session = _Session()
    box = {
        "home_id": "T1",
        "away_id": "T2",
        "home_goals": 4,
        "away_goals": 1,
        "overtime": False,
        "scoring_events": [
            {"scorer_id": "p_test_1", "scorer": "Alex Example", "for_team_id": "T1"},
            {"scorer_id": "p_test_1", "scorer": "Alex Example", "for_team_id": "T1"},
            {"scorer_id": "p_test_1", "scorer": "Alex Example", "for_team_id": "T1"},
            {"scorer_id": "p_other", "scorer": "Other", "for_team_id": "T1"},
        ],
    }
    emitted = ingest_game_box_storylines(session, box)
    assert emitted >= 1
    headlines = [str(s.get("headline") or "") for s in session.storyline_events]
    assert any("hat trick" in h.lower() for h in headlines)
    graph = session.knowledge_graph
    assert any(row.get("headline") for row in graph)
    log = session.player_recent_games.get("p_test_1") or []
    assert log and int(log[-1]["g"]) >= 3


def test_coverage_payload_has_dossiers_and_graph():
    session = _Session()
    session.knowledge_graph = [
        {
            "headline": "Camp restless",
            "knowledge_type": "claim",
            "public_knowledge_level": "rumour",
            "player_name": "Alex Example",
        }
    ]
    payload = coverage_payload_fields(session)
    assert payload["player_dossiers"]
    dossier = payload["player_dossiers"][0]
    assert dossier["player_name"] == "Alex Example"
    assert "wants" in dossier and "trusts" in dossier and "remembers" in dossier
    assert payload["insider_items"]
    assert payload["beat_writers"]


def test_breakup_mutates_life_state():
    import random
    from app.sim_engine.franchise.storyline_engine import _u_mutate_life_from_event  # noqa: E402

    entity = {
        "player_id": "p1",
        "life": {"relationship_status": "partnered", "dependents": 0, "partner": {"id": "fam_p1_partner"}},
        "state": {},
    }
    spec = {"id": "breakup", "event_tier": "meaningful"}
    _u_mutate_life_from_event(entity, spec, random.Random(1), day=40)
    assert entity["life"]["relationship_status"] == "single"
    assert "partner" not in entity["life"]


def test_new_child_increments_dependents_and_creates_child():
    import random
    from app.sim_engine.franchise.storyline_engine import _u_mutate_life_from_event  # noqa: E402

    entity = {
        "player_id": "p2",
        "life": {"relationship_status": "married", "dependents": 1, "children": [{"id": "c0", "age_bracket": "toddler"}]},
        "state": {},
    }
    spec = {"id": "new_child", "event_tier": "major"}
    _u_mutate_life_from_event(entity, spec, random.Random(2), day=50)
    assert entity["life"]["dependents"] == 2
    assert len(entity["life"]["children"]) == 2
    assert entity["life"]["children"][-1]["age_bracket"] == "infant"


def test_human_dossier_payload_tier_language():
    from app.sim_engine.franchise.storyline_engine import build_human_dossier_payload  # noqa: E402

    session = _Session()
    entity = {
        "player_id": "p_test_1",
        "overall": 82,
        "personality": {
            "character": 78,
            "competitiveness": 92,
            "professionalism": 85,
            "loyalty": 79,
            "volatility": 25,
            "family_orientation": 72,
            "resilience": 80,
            "sociability": 60,
            "money_focus": 40,
            "ambition": 88,
        },
        "state": {"morale": 62, "confidence": 58, "role_satisfaction": 34, "gm_trust": 41},
        "life": {"relationship_status": "married", "dependents": 2, "city_attachment": 70, "home_stability": 75, "relocation_strain": 18},
        "human_pressure": {"tier": 2, "tier_label": "Frustrated", "drivers": [{"label": "Role", "value": 12}]},
        "mental_wellbeing": {"state": "strained", "wellbeing_score": 58},
    }
    dossier = build_human_dossier_payload(session, entity, include_private=True)
    assert dossier["character"]["headline"] in ("Strong", "High", "Very High", "Above Average", "Elite")
    assert dossier["current_state"]["base_ovr"] == 82.0
    assert dossier["current_state"]["pressure_label"] == "Frustrated"
    assert dossier["life"]["summary"].startswith("Married")
    assert dossier["mental_wellbeing"]["private"] is True


def test_human_pressure_competing_forces():
    from app.sim_engine.franchise.storyline_engine import _u_compute_human_pressure  # noqa: E402

    loyal_winner = {
        "personality": {"competitiveness": 92, "loyalty": 79, "volatility": 30, "resilience": 80},
        "state": {"role_satisfaction": 34, "gm_trust": 65, "coach_trust": 60, "belonging": 62, "personal_stress": 25, "media_stress": 20},
        "life": {"city_attachment": 60, "relocation_strain": 15},
        "concerns": {"winning": {"satisfaction": 78}},
        "memories": [],
    }
    volatile_loser = {
        "personality": {"competitiveness": 70, "loyalty": 45, "volatility": 82, "resilience": 48, "ego": 81},
        "state": {"role_satisfaction": 38, "gm_trust": 41, "coach_trust": 44, "belonging": 40, "personal_stress": 35, "media_stress": 40},
        "life": {"city_attachment": 32, "relocation_strain": 45},
        "concerns": {"winning": {"satisfaction": 28}},
        "memories": [{"kind": "betrayal", "emotional_delta": 8, "calendar_day": 10}],
    }
    p_loyal = _u_compute_human_pressure(loyal_winner)
    p_volatile = _u_compute_human_pressure(volatile_loser)
    assert p_volatile["score"] > p_loyal["score"]
    assert p_volatile["tier"] >= p_loyal["tier"]


def test_player_meetings_catalog_and_ovr_explanation():
    from app.sim_engine.franchise.storyline_engine import (  # noqa: E402
        build_ovr_trend_explanation,
        get_available_gm_interactions,
        _gm_relationship_summary,
    )

    session = _Session()
    entity = session.universe_players["p_test_1"]
    entity["state"] = {
        "morale": 42,
        "confidence": 38,
        "gm_trust": 35,
        "role_satisfaction": 32,
        "personal_stress": 55,
    }
    entity["personality"] = {"loyalty": 55, "ambition": 80, "volatility": 60}
    rel = _gm_relationship_summary(entity, session, "p_test_1")
    assert rel["label"] in ("Neutral", "Strained", "Broken")
    ovr = build_ovr_trend_explanation(session, "p_test_1")
    assert "factors" in ovr
    avail = get_available_gm_interactions(session, "p_test_1")
    assert isinstance(avail, list)
    assert len(avail) >= 5
    ids = {row["id"] for row in avail}
    assert "discuss_ice_time" in ids or "repair_relationship" in ids
    assert "ask_ntc_waiver" not in ids


def test_default_traits_get_diversified_and_custom_traits_stay():
    from app.sim_engine.engine import diversify_player_personality_and_psych, ensure_player_character_initialized
    from app.sim_engine.entities.player import PersonalityTraits
    from app.sim_engine.franchise.storyline_engine import _u_personality_tags, _u_tier_label

    class _Flat:
        id = "flat_a"
        character = 70
        traits = PersonalityTraits()
        psych = None

    rng = __import__("random").Random(1)
    ensure_player_character_initialized(_Flat, rng)
    axes = [getattr(_Flat.traits, k) for k in ("loyalty", "ego", "volatility", "ambition", "leadership")]
    assert max(axes) - min(axes) > 0.12
    pmap = {
        "character": 70,
        "ego": _Flat.traits.ego * 100,
        "ambition": _Flat.traits.ambition * 100,
        "volatility": _Flat.traits.volatility * 100,
        "resilience": 70,
        "professionalism": 70,
        "leadership": _Flat.traits.leadership * 100,
        "media_savvy": 40,
        "sociability": 50,
        "coachability": 60,
        "family_orientation": _Flat.traits.family_priority * 100,
        "money_focus": _Flat.traits.money_focus * 100,
        "competitiveness": _Flat.traits.competitiveness * 100,
        "clutch": 50,
    }
    tags = _u_personality_tags({"personality": pmap})
    assert tags
    custom = _Player()
    before = custom.traits.loyalty
    diversify_player_personality_and_psych(custom, rng)
    assert custom.traits.loyalty == before
    assert _u_tier_label(22) == "Very Low"
    assert _u_tier_label(88) != "Average"


def test_stats_ledger_box_moves_morale():
    from app.sim_engine.franchise.storyline_engine import apply_universe_postgame
    from app.sim_engine.franchise.storyline_stat_bridge import player_stat_map_from_box

    class _Team:
        roster = [_Player()]

    session = _Session()
    session.team_by_id = {"T1": _Team()}
    session.universe_players["p_test_1"]["state"] = {"morale": 55.0, "confidence": 55.0, "media_stress": 20.0}
    box = {
        "home_id": "T1",
        "away_id": "T2",
        "home_goals": 4,
        "away_goals": 1,
        "home_skaters": [
            {"player_id": "p_test_1", "g": 2, "a": 1, "sog": 5, "toi_sec": 1100, "pim": 0},
        ],
    }
    stats = player_stat_map_from_box(box)
    assert stats["p_test_1"]["points"] == 3
    apply_universe_postgame(session, "T1", {"won": True, "player_stats": stats, "game_id": "g1"})
    morale = float(session.universe_players["p_test_1"]["state"]["morale"])
    assert morale > 60
    assert session.team_by_id["T1"].roster[0].psych.morale > 0.55


