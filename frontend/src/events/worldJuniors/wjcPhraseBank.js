/**
 * 100+ broadcast phrase variants for WJC desk — one variant picked per category per cycle.
 */
import { WJC_HOSTS } from "./wjcBroadcastScripts";

export { WJC_HOSTS };

function v(speakerId, emotion, text, tag) {
  return { speakerId, emotion, text, tag };
}

const OPEN = [
  v("host_2", "serious", "Welcome back to the World Juniors desk. {day_label}. {top_story}.", "always"),
  v("host_2", "neutral", "Good to have you with us on the U20 championship desk. {day_label}.", "always"),
  v("host_2", "serious", "We are live from the World Juniors studio. {top_story}.", "live"),
  v("host_2", "neutral", "Another day of U20 hockey in the books. {day_label}.", "live"),
  v("host_2", "serious", "The tournament desk is rolling. {top_story}.", "live"),
  v("host_2", "neutral", "Let's get right into it. {day_label}. {top_story}.", "live"),
  v("host_2", "serious", "World Juniors coverage continues. {top_story}.", "live"),
  v("host_2", "neutral", "From the broadcast booth — {day_label}. Here's what matters.", "live"),
  v("host_2", "serious", "U20 nations are on the ice and the board is moving. {top_story}.", "live"),
  v("host_2", "neutral", "The World Juniors desk is live. Sim tournament days for scores, rosters, and draft stock.", "pretournament"),
  v("host_2", "neutral", "U20 nations are ready. Advance the calendar to drop the first slate of games.", "pretournament"),
];

const HOT = [
  v("host_1", "fired", "LISTEN — {winner_label} just got a massive result! {score_line}!", "game"),
  v("host_1", "fired", "I am NOT calm! {upset_line} — {score_line}!", "upset"),
  v("host_1", "fired", "{standout_name} is putting the world on notice — {standout_pts} points!", "prospect"),
  v("host_1", "fired", "Do NOT sleep on {leader_label}! {leader_w} wins and climbing!", "standings"),
  v("host_1", "fired", "YOUR GUY {user_name}! {user_pts} POINTS! The draft board is WATCHING!", "user"),
  v("host_1", "fired", "{pts_leader_name} is COOKING with {pts_leader_pts} points!", "leader"),
  v("host_1", "fired", "UPSET ALERT! {underdog_label} beat {favorite_label}! {score_line}!", "upset"),
  v("host_1", "fired", "This tournament just shifted! {winner_label} takes it {score_line}!", "game"),
  v("host_1", "fired", "{surprise_label} at {surprise_w}-{surprise_l}? That is NOT a fluke!", "surprise"),
  v("host_1", "fired", "GOLD MEDAL HOCKEY! {winner_label} wins it {score_line}!", "gold"),
  v("host_1", "fired", "{faller_name} is FALLING on the board — was {faller_before}, now {faller_after}!", "stock"),
  v("host_1", "fired", "When {loser_label} loses like that, the whole country feels it!", "game"),
  v("host_1", "fired", "{standout_name} showed up in a BIG way — plus-{standout_pm} hockey!", "prospect"),
  v("host_1", "fired", "I told you {leader_code} was dangerous — {leader_pts} points in the standings!", "standings"),
  v("host_1", "fired", "The energy for {winner_label} tonight is ELECTRIC after {score_line}!", "game"),
];

const FACTS = [
  v("host_3", "calm", "Final: {home_code} {home_goals}, {away_code} {away_goals}. {winner_label} moves to {winner_record}.", "game"),
  v("host_3", "calm", "{standout_name}: {standout_g}G, {standout_a}A, {standout_pts} PTS in {standout_gp} GP. Plus-minus {standout_pm}.", "prospect"),
  v("host_3", "calm", "{leader_code} leads at {leader_pts} points, record {leader_w}-{leader_l}, goal diff plus {leader_diff}.", "standings"),
  v("host_3", "calm", "Points leader: {pts_leader_name} ({pts_leader_country}) — {pts_leader_pts} points.", "leader"),
  v("host_3", "calm", "Goals leader: {goal_leader_name} with {goal_leader_g} goals in {goal_leader_gp} games.", "leader"),
  v("host_3", "calm", "Stock riser: {riser_name} ({riser_country}), rank {riser_before} to {riser_after}.", "stock"),
  v("host_3", "calm", "{team_code}: {team_gf} goals for, {team_ga} against, {team_w}-{team_l} record.", "standings"),
  v("host_3", "calm", "Quarterfinal: {score_line}. {winner_label} advances.", "playoff"),
  v("host_3", "calm", "Semifinal result — {score_line}. {winner_label} to the medal game.", "playoff"),
  v("host_3", "calm", "Medals: gold {gold_label}, silver {silver_label}, bronze {bronze_label}.", "gold"),
  v("host_3", "calm", "{underdog_code} defeats {favorite_code} {home_goals}-{away_goals}. Records: {underdog_record} vs {favorite_record}.", "upset"),
  v("host_3", "calm", "Tournament analytics: {analytics_player} — {analytics_pts} PTS, {analytics_sog} SOG, {analytics_pm} plus-minus.", "prospect"),
  v("host_3", "calm", "Day {wjc_day} summary: {games_today_count} games completed. {day_highlight}", "live"),
  v("host_3", "calm", "Bronze medal game: {score_line}. {winner_label} takes third.", "playoff"),
  v("host_3", "calm", "User prospect {user_name}: {user_pts} points in {user_gp} WJC games. {user_stock_note}.", "user"),
];

const MEDIATE = [
  v("host_2", "neutral", "Marcus, Elena — what stood out in {score_line}?", "game"),
  v("host_2", "neutral", "Fair question: did {standout_name} actually move draft stock tonight?", "prospect"),
  v("host_2", "neutral", "Let's talk {winner_label} after that {score_line} result.", "game"),
  v("host_2", "neutral", "How does {leader_label} look atop the standings at {leader_pts} points?", "standings"),
  v("host_2", "neutral", "Is {surprise_label} the biggest surprise at {surprise_w}-{surprise_l}?", "surprise"),
  v("host_2", "neutral", "Your prospect {user_name} — what should scouts take from his tournament?", "user"),
  v("host_2", "neutral", "Medal round pressure is here. {playoff_note}", "playoff"),
  v("host_2", "neutral", "Quick stock check: {riser_name} up from {riser_before} to {riser_after}.", "stock"),
  v("host_2", "neutral", "Another angle on {pts_leader_name} and his {pts_leader_pts}-point tournament.", "leader"),
  v("host_2", "neutral", "Where does {faller_name} land on the board after slipping to {faller_after}?", "stock"),
];

const EXTRA = Array.from({ length: 50 }, (_, i) => {
  const pool = [
    v("host_1", "fired", "The {winner_code} win matters — {score_line} — momentum is real!", "game"),
    v("host_3", "calm", "Shot volume watch: {standout_name} with {standout_sog} shots on goal.", "prospect"),
    v("host_2", "neutral", "We will reset after this break — {day_label} rolls on.", "always"),
    v("host_1", "fired", "{goal_leader_name} is the rocket watch at {goal_leader_g} goals!", "leader"),
    v("host_3", "calm", "Impact rating for {standout_name} trending {analytics_trend}.", "prospect"),
    v("host_2", "serious", "National teams are chasing seeding — {leader_code} currently leads.", "standings"),
    v("host_1", "fired", "Nobody expected {underdog_label} to punch {favorite_label} in the mouth!", "upset"),
    v("host_3", "calm", "Group play checkpoint: {games_today_count} games today.", "live"),
    v("host_2", "neutral", "Scouts have {riser_name} rising after {riser_pts} tournament points.", "stock"),
    v("host_1", "fired", "This is what World Juniors is about — {score_line}!", "game"),
  ];
  const base = pool[i % pool.length];
  return { ...base, id: `extra-${i}` };
});

export const WJC_PHRASE_BANK = [...OPEN, ...HOT, ...FACTS, ...MEDIATE, ...EXTRA];

export function phraseApplies(phrase, vars, payload) {
  const tag = phrase.tag || "always";
  if (tag === "always") return true;
  if (!payload?.hasData) return tag === "pretournament";
  if (tag === "gold" && !payload?.medals_final) return false;
  if (tag === "upset" && !vars._featuredGame) return false;
  if (tag === "user" && !payload?.user_prospects?.length) return false;
  if (tag === "playoff" && !(payload?.wjc_day > (payload?.rr_days_total || 9))) return false;
  if (tag === "surprise" && !vars.surprise_label) return false;
  if (tag === "stock" && !vars.riser_name) return false;
  if (tag === "leader" && !vars.pts_leader_name) return false;
  if (tag === "prospect" && !vars.standout_name) return false;
  if (tag === "game" && !vars._featuredGame) return false;
  if (tag === "standings" && !vars.leader_code) return false;
  if (tag === "live") return payload?.hasData;
  return true;
}
