"""PuckGPT history questions: routing for the long-form questions, and the judging rules on a tiny league."""

from types import SimpleNamespace as NS

import pytest

from services.puckgpt import history as H

CASES = [
    ('Which trade in franchise history produced the most total WAR/value for us?', 'best_trade_for_us'),
    ('What was the biggest mistake this organization ever made?', 'mistake'),
    ('Which player did we give up on too early?', 'gave_up_early'),
    ('Which draft pick hurt us the most because of who was selected afterward?', 'pick_regret'),
    ('Which GM decision had the largest negative impact over a 10-year span?', 'mistake'),
    ('Which season unexpectedly started our dynasty?', 'dynasty_start'),
    ('Which player ended up becoming the face of the franchise when nobody expected it?', 'face_unexpected'),
    ('Which trade looked terrible at the time but ended up helping us?', 'terrible_then_good'),
    ('Which move officially ended our championship window?', 'window_end'),
    ('Which player generated the most value relative to acquisition cost?', 'value_vs_cost'),
    ('What is the biggest draft steal in franchise history?', 'draft_steal'),
    ('Which draft class produced the most NHL games?', 'class_totals'),
    ('Which draft class produced the most total points?', 'class_totals'),
    ('Which first-round pick was our biggest bust?', 'first_round_bust'),
    ('Which late-round pick became the best NHL player?', 'draft_steal'),
    ('Which prospect exceeded scouting expectations the most?', 'scouting_beat'),
    ('Which prospect failed despite being considered untouchable?', 'scouting_beat'),
    ('Which draft produced the highest number of future award winners?', 'class_awards'),
    ('Which player had the largest gap between draft position and career success?', 'gap_draft_success'),
    ('What was the strongest draft class in league history?', 'strongest_class'),
    ('Which trade is considered the greatest robbery in league history?', 'robbery'),
    ('What trade changed the balance of power in the NHL the most?', 'balance_of_power'),
    ('Which team benefited most from trading with us?', 'benefited_from_us'),
    ('Which trade deadline acquisition had the biggest playoff impact?', 'deadline_impact'),
    ('Which player has been traded the most times?', 'traded_most'),
    ('Which superstar changed teams most unexpectedly?', 'superstar_moved'),
    ('Which trade offer did we reject that would have been incredible in hindsight?', 'rejected_offer'),
    ('Which team has historically won the most trades?', 'team_trade_record'),
    ('Which trade directly led to a Stanley Cup?', 'trade_to_cup'),
    ('Which trade asset was the most frequently exchanged pick or player?', 'most_exchanged'),
    ('What was the biggest playoff upset in league history?', 'upset'),
    ('Which Stanley Cup champion had the weakest regular-season record?', 'weakest_champion'),
    ('Which team blew the biggest series lead ever?', 'blown_lead'),
    ('Which player scored the biggest goal in franchise history?', 'biggest_goal'),
    ('Which playoff run was the most improbable?', 'improbable_run'),
    ('Which goalie carried a team furthest in the playoffs?', 'goalie_carry'),
    ('Which player has the most Game 7 goals?', 'game7_goals'),
    ('What was the closest Cup Final ever played?', 'closest_final'),
    ('Which playoff team overachieved most relative to projections?', 'overachieved'),
    ('Which franchise suffered the most heartbreaking elimination?', 'heartbreak'),
    ('Which player had the strangest career path?', 'strangest_path'),
    ('Which player improved the most after leaving our organization?', 'gave_up_early'),
    ('Which player declined the fastest after reaching stardom?', 'declined_fastest'),
    ('Who was the greatest one-season wonder?', 'one_season_wonder'),
    ('Which player had the best career despite never winning a major award?', 'best_no_award'),
    ('Which player had the longest NHL career?', 'longest_career'),
    ('Which player experienced the biggest overall-rating increase in history?', 'biggest_rise'),
    ('Which player had the most unexpected breakout season?', 'breakout'),
    ('Which player was consistently underrated throughout his career?', 'underrated_career'),
    ('Which retired player would most improve our current roster if brought back in his prime?', 'retired_prime'),
    ('What would league history look like if we never made the Smith trade in 2031?', 'counterfactual'),
    ('Which single event altered league history the most?', 'single_event'),
    ('Which injury had the biggest ripple effect on future championships?', 'injury_ripple'),
    ('Which prospect becoming a bust changed the standings the most?', 'bust_standings'),
    ("Which team's dynasty was closest to never happening?", 'dynasty_close'),
    ('Which franchise has the most alternate-history "what if" moments?', 'most_what_ifs'),
    ('What move created the largest butterfly effect in the save?', 'butterfly'),
    ('Which Cup winner would likely not have won without a specific trade?', 'trade_to_cup'),
    ('Which draft lottery outcome changed the future most dramatically?', 'lottery'),
    ('Which player is remembered as the biggest "what could have been" story?', 'what_could_have_been'),
    ("What's the weirdest record still standing?", 'weird_record'),
    ('Which player owns the most unbreakable record?', 'unbreakable'),
    ('What is the highest-scoring game ever played?', 'high_scoring_game'),
    ('What is the longest game in league history?', 'longest_game'),
    ('Which season featured the most chaos?', 'chaos'),
    ('Which team had the greatest single-season collapse?', 'collapse'),
    ('Which expansion franchise became competitive the fastest?', 'expansion'),
    ('Which player unexpectedly became a Hall of Famer?', 'unexpected_hof'),
    ('Which season produced the most league-wide talent?', 'season_talent'),
    ('Which team came closest to perfection?', 'perfection'),
    ('Who is the forgotten legend of this save?', 'forgotten'),
    ('Which player had Hall-of-Fame talent but terrible luck?', 'hof_bad_luck'),
    ("Which player's legacy is most misunderstood?", 'forgotten'),
    ('Who benefited the most from being drafted by their specific organization?', 'org_benefit'),
    ('Which star player is least remembered despite elite production?', 'forgotten'),
    ('Which championship team is most likely to be forgotten by history?', 'forgettable_champion'),
    ('Who is the greatest player never to win a Cup?', 'forgotten'),
    ('Which franchise has wasted the most talent?', 'wasted_talent'),
    ('Which player defined an entire era of the league?', 'forgotten'),
    ('If fans could change one event in league history, what would it be?', 'fans_change'),
]


# Named CPU clubs and loose wording reach the same handlers.
CASES += [
    ("What was the biggest mistake the Vancouver Canucks ever made?", "mistake"),
    ("Which team benefited most from trading with Vancouver?", "benefited_from_us"),
    ("Which move officially ended the Golden Knights championship window?", "window_end"),
    ("Which season started the Golden Knights' best run?", "dynasty_start"),
    ("Which player became the face of the Columbus franchise when nobody expected it?", "face_unexpected"),
    ("wat was the canucks biggest screw up", "mistake"),
    ("what was are biggest screw up", "mistake"),
    ("whos the guy the flames shouldnt of traded", "gave_up_early"),
    ("habs biggest mistake", "mistake"),
]


@pytest.mark.parametrize("q,handler", CASES)
def test_history_questions_route(q, handler):
    fn = H.match(q)
    assert fn is not None and fn.__name__ == handler, (q, fn and fn.__name__)


@pytest.mark.parametrize("q", ["who is my best defenseman", "most goals this season", "standings", "top prospects", "my trades"])
def test_everyday_questions_are_left_alone(q):
    assert H.match(q) is None


def _hist():
    seasons = {
        "2026": {"standings": {"A": {"pts": 110, "gp": 82}, "B": {"pts": 80, "gp": 82}}, "champion_id": "B",
                 "playoff_series": [{"round_index": 1, "team_high_id": "A", "team_low_id": "B", "wins_high": 3, "wins_low": 4, "winner_id": "B",
                                     "games": [{"game": i, "home_id": "A", "away_id": "B", "home_score": s[0], "away_score": s[1],
                                                "winner_id": "A" if s[0] > s[1] else "B", "ot_periods": 3 if i == 7 else 0,
                                                "goals": [["B", "p2", 4]] if i == 7 else [], "gwg": "p2" if i == 7 else None}
                                               for i, s in enumerate([(3, 1), (2, 1), (4, 0), (1, 2), (0, 2), (1, 3), (1, 2)], 1)]}]},
    }
    players = {
        "p1": {"name": "Ace", "position": "C", "seasons": [{"season": 2026, "team_id": "B", "source": "simulated", "gp": 70, "pts": 60}]},
        "p2": {"name": "Bee", "position": "C", "seasons": [{"season": 2026, "team_id": "A", "source": "simulated", "gp": 70, "pts": 20}]},
    }
    trades = {"t1": {"trade_id": "t1", "season": 2026, "iso": "2026-11-01", "teams": ["A", "B"], "picks": [],
                     "players": [{"player_id": "p1", "name": "Ace", "to_team_id": "B"}, {"player_id": "p2", "name": "Bee", "to_team_id": "A"}],
                     "original_valuation": {"A": 50, "B": 40}}}
    session = NS(legacy={"seasons": seasons, "players": players, "trades": trades}, season_phase="offseason", draft_results_archive=[])
    view = NS(session=session, user_tid="A", season=2027, players={}, team_meta={"A": {"abbr": "AAA", "name": "Alphas"}, "B": {"abbr": "BBB", "name": "Betas"}},
              row=lambda pid: {}, team_rows=lambda tid: [])
    h = H.Hist.__new__(H.Hist)
    h.view, h.session, h.lg, h.user = view, session, session.legacy, "A"
    h.seasons = {int(k): v for k, v in seasons.items()}
    h.players, h.live, h._lines, h._trades, h._drafts = players, False, {}, None, {}
    return h


def test_trades_are_judged_on_what_each_side_got_after():
    h = _hist()
    t = h.trades()[0]
    assert H.Hist.net(t, "B") == 40 and h.winner(t) == "B"
    assert "BBB came out +40" in H.robbery(h, "")["answer"]
    assert "Betas" in H.benefited_from_us(h, "")["answer"]


def test_playoff_games_give_blown_leads_game7_goals_and_overtime():
    h = _hist()
    assert "up 3 games" in H.blown_lead(h, "")["answer"]
    assert H.game7_goals(h, "")["answer"].startswith("Bee: 1 Game 7 goal")
    assert "3rd overtime" in H.longest_game(h, "")["answer"]
    assert "who had 30 more points, in 7 games" in H.upset(h, "")["answer"]


def test_cpu_club_answers_read_in_third_person():
    out = H._third_person({"answer": "You haven't made a trade; it cost you 5 and left you short. Your pick.", "analysis": ["Value with you"]},
                          "Vancouver Canucks")
    assert out["answer"] == "The Vancouver Canucks haven't made a trade; it cost them 5 and left them short. Their pick."
    assert out["analysis"] == ["Value with them"]


def test_named_club_reads_history_from_its_side():
    h = H._for_team(_hist(), "B")
    assert H.best_trade_for_us(h, "franchise")["answer"].endswith("Net +40 value for you.")
