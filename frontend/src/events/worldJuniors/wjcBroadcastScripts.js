/**
 * ~30 broadcast templates for the WJC desk trio.
 * host_1 = left hot-take analyst (SAS energy)
 * host_2 = center mediator
 * host_3 = right facts desk
 */

export const WJC_HOSTS = {
  host_1: { id: "host_1", name: "Marcus Cole", role: "Hot Take Analyst", side: "left" },
  host_2: { id: "host_2", name: "Jordan Hayes", role: "Lead Anchor", side: "center" },
  host_3: { id: "host_3", name: "Dr. Elena Park", role: "Scouting Desk", side: "right" },
};

export const WJC_SCRIPT_TEMPLATES = [
  {
    id: "open_desk",
    speakerId: "host_2",
    emotion: "serious",
    text: "Welcome back to the World Juniors desk. {day_label}. {top_story}.",
  },
  {
    id: "hot_open",
    speakerId: "host_1",
    emotion: "fired",
    text: "LISTEN — {upset_line} I am NOT calm about this! {score_line} changes EVERYTHING for {winner_label}!",
  },
  {
    id: "facts_open",
    speakerId: "host_3",
    emotion: "calm",
    text: "By the numbers: {home_code} {home_goals}, {away_code} {away_goals}. {winner_label} improves to {winner_record}. {loser_label} sits at {loser_record}.",
  },
  {
    id: "mediator_question",
    speakerId: "host_2",
    emotion: "neutral",
    text: "Fair question for both of you — what did {standout_name} do tonight that actually moved draft stock?",
  },
  {
    id: "prospect_explosion",
    speakerId: "host_1",
    emotion: "fired",
    text: "{standout_name} just put up {standout_pts} points for {standout_country}! Scouts are LOSING THEIR MINDS — stock was {stock_before}, now trending toward {stock_after}!",
  },
  {
    id: "prospect_facts",
    speakerId: "host_3",
    emotion: "calm",
    text: "{standout_name}: {standout_g} goals, {standout_a} assists, {standout_gp} GP, plus-minus {standout_pm}. Tournament total: {standout_pts} points. Draft rank moved from {stock_before} to {stock_after}.",
  },
  {
    id: "standings_lead",
    speakerId: "host_3",
    emotion: "calm",
    text: "{leader_code} leads the group at {leader_pts} points with a {leader_w}-{leader_l} record. Goal differential: plus {leader_diff}.",
  },
  {
    id: "standings_hot",
    speakerId: "host_1",
    emotion: "fired",
    text: "Do NOT sleep on {leader_label}! {leader_w} wins! They are playing like the favorite and everybody else is on NOTICE!",
  },
  {
    id: "upset_alert",
    speakerId: "host_1",
    emotion: "fired",
    text: "UPSET ALERT! {underdog_label} took down {favorite_label} {score_line}! That is a NATIONAL PROGRAM statement!",
  },
  {
    id: "upset_facts",
    speakerId: "host_3",
    emotion: "calm",
    text: "Upset confirmed. {underdog_code} defeats {favorite_code} {score_line}. {underdog_code} record: {underdog_record}. {favorite_code} drops to {favorite_record}.",
  },
  {
    id: "user_prospect",
    speakerId: "host_2",
    emotion: "neutral",
    text: "Your franchise prospect {user_name} is on {user_country_label}. Tournament line: {user_pts} points in {user_gp} games. Stock trajectory: {user_stock_note}.",
  },
  {
    id: "user_prospect_hot",
    speakerId: "host_1",
    emotion: "fired",
    text: "YOUR GUY {user_name}! {user_pts} POINTS! {user_country_label} needs him and the draft board is WATCHING!",
  },
  {
    id: "goal_leader",
    speakerId: "host_3",
    emotion: "calm",
    text: "Tournament goals leader: {goal_leader_name} ({goal_leader_country}) with {goal_leader_g} goals in {goal_leader_gp} games.",
  },
  {
    id: "points_leader",
    speakerId: "host_3",
    emotion: "calm",
    text: "Points leader: {pts_leader_name} — {pts_leader_pts} points ({pts_leader_g}G, {pts_leader_a}A) for {pts_leader_country}.",
  },
  {
    id: "points_hot",
    speakerId: "host_1",
    emotion: "fired",
    text: "{pts_leader_name} is COOKING! {pts_leader_pts} points! That is first-line production on the WORLD STAGE!",
  },
  {
    id: "playoff_preview",
    speakerId: "host_2",
    emotion: "serious",
    text: "Medal round pressure is here. {playoff_note} The quarterfinal picture is taking shape.",
  },
  {
    id: "qf_result",
    speakerId: "host_3",
    emotion: "calm",
    text: "Quarterfinal final: {score_line}. {winner_label} advances. {loser_label} is eliminated from gold contention.",
  },
  {
    id: "sf_result",
    speakerId: "host_3",
    emotion: "calm",
    text: "Semifinal: {score_line}. {winner_label} moves on to the medal game.",
  },
  {
    id: "bronze_medal",
    speakerId: "host_2",
    emotion: "serious",
    text: "Bronze medal game: {score_line}. {winner_label} takes third place.",
  },
  {
    id: "gold_medal",
    speakerId: "host_1",
    emotion: "fired",
    text: "GOLD MEDAL HOCKEY! {score_line}! {winner_label} ARE WORLD JUNIOR CHAMPIONS!",
  },
  {
    id: "gold_facts",
    speakerId: "host_3",
    emotion: "calm",
    text: "Gold medal final: {score_line}. {gold_label} wins gold. {silver_label} silver. {bronze_label} bronze.",
  },
  {
    id: "stock_riser",
    speakerId: "host_3",
    emotion: "calm",
    text: "Biggest stock riser on our board: {riser_name} ({riser_country}). Rank {riser_before} → {riser_after}. Tournament: {riser_pts} points.",
  },
  {
    id: "stock_faller",
    speakerId: "host_1",
    emotion: "fired",
    text: "And {faller_name} is FALLING on the board! Was {faller_before}, now {faller_after}! {faller_country} needed more and did not get it!",
  },
  {
    id: "team_offense",
    speakerId: "host_3",
    emotion: "calm",
    text: "{team_code} has scored {team_gf} goals in {team_gp} games while allowing {team_ga}. Record: {team_w}-{team_l}.",
  },
  {
    id: "impact_win",
    speakerId: "host_3",
    emotion: "calm",
    text: "Win impact: {standout_name} posted plus-{standout_pm} in a {winner_code} victory. Analytics favor players who drive positive goal share in wins.",
  },
  {
    id: "impact_loss",
    speakerId: "host_1",
    emotion: "fired",
    text: "When you LOSE like {loser_label} did, guys like {standout_name} still show up — but minus hockey does NOT help draft stock!",
  },
  {
    id: "day_recap",
    speakerId: "host_2",
    emotion: "neutral",
    text: "Day {wjc_day} recap: {games_today_count} games on the board. {day_highlight}",
  },
  {
    id: "ticker_hook",
    speakerId: "host_2",
    emotion: "neutral",
    text: "Stay with us — scores rolling on the ticker. Next question: which nation is the biggest surprise of the tournament so far?",
  },
  {
    id: "surprise_nation",
    speakerId: "host_1",
    emotion: "fired",
    text: "It's {surprise_label}! Nobody had them playing this well at {surprise_w}-{surprise_l}!",
  },
  {
    id: "analytics_close",
    speakerId: "host_3",
    emotion: "calm",
    text: "Closing data point: {analytics_player} — {analytics_pts} points, {analytics_sog} shots, {analytics_pm} plus-minus across {analytics_gp} GP. Impact rating trending {analytics_trend}.",
  },
  {
    id: "pretournament",
    speakerId: "host_2",
    emotion: "neutral",
    text: "The World Juniors desk is live. Tournament action is loading — sim the next day to drop fresh scores, prospect lines, and draft stock movement.",
  },
];
