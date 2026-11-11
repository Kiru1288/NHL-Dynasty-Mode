"""Booth stories for the live sim (SimTracker): data + a small selector.

Two-person "radio booth" exchanges told in lulls. Every template belongs to a category;
a category's facts are computed from the live franchise session (never invented) and a
template is only eligible when every slot it uses is filled by a real fact.

Template shape per language/category: {"s": [setup lines], "p": [payoff lines], "r": [reaction lines]}.
A told exchange = one setup + one payoff (+ one reaction). Composed template id = "cat:i:j".
Real-world history is limited to stable facts (first NHL season, previous city, Stanley Cup count).
"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional, Tuple

# ── stable real-world franchise facts: (first NHL season of the franchise, previous city or None, Stanley Cups)
TEAM_HISTORY: Dict[str, Tuple[int, Optional[str], int]] = {
    "ANA": (1993, None, 1), "BOS": (1924, None, 6), "BUF": (1970, None, 0), "CGY": (1972, "Atlanta", 1),
    "CAR": (1979, "Hartford", 1), "CHI": (1926, None, 6), "COL": (1979, "Quebec", 3), "CBJ": (2000, None, 0),
    "DAL": (1967, "Minnesota", 1), "DET": (1926, None, 11), "EDM": (1979, None, 5), "FLA": (1993, None, 2),
    "LAK": (1967, None, 2), "MIN": (2000, None, 0), "MTL": (1917, None, 24), "NSH": (1998, None, 0),
    "NJD": (1974, "Colorado", 3), "NYI": (1972, None, 4), "NYR": (1926, None, 4), "OTT": (1992, None, 0),
    "PHI": (1967, None, 2), "PIT": (1967, None, 5), "SJS": (1991, None, 0), "SEA": (2021, None, 0),
    "STL": (1967, None, 1), "TBL": (1992, None, 3), "TOR": (1917, None, 13), "UTA": (2024, None, 0),
    "VAN": (1970, None, 0), "VGK": (2017, None, 1), "WSH": (1974, None, 1), "WPG": (1999, "Atlanta", 0),
}

_PLURAL_SLOTS = frozenset({"g", "a", "pts", "gp", "n", "games", "ppg", "hits", "blk", "w", "l", "delta", "gap", "cups", "sv", "sa", "o"})
_ORD_EN = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

# ════════════════════════════════════════════════════════════════════ ENGLISH
EN: Dict[str, Dict[str, List[str]]] = {
    # ── sim data
    "team_leader": {
        "s": ["Who's been driving the bus for {team} this year?", "Let's talk about the {team} offence for a second.",
              "If you want one name on this {team} roster, who is it?", "Somebody has to carry the scoring load for {team}.", "Quick look at the {team} scoring sheet."],
        "p": ["That's {player}. {pts} points through {gp} games, {g} of them goals.", "{player} leads the team with {pts} points, {g} goals and {a} assists.",
              "It's {player}, and it isn't close. {pts} points in {gp} games.", "{player}. {g} goals, {a} helpers, {pts} points to lead the club."],
        "r": ["That's the guy they build every power play around.", "You notice him every shift, and the numbers back it up.", "Hard to argue with that production."],
    },
    "goal_pace": {
        "s": ["Here's a number that jumped out at me today.", "I was doing some math in the hotel this afternoon.",
              "You want a pace stat? I've got one.", "Let me throw a projection at you."],
        "p": ["{player} has {g} goals in {gp} games. That's a {pace}-goal pace over a full season.", "{player} is on pace for {pace} goals, sitting at {g} through {gp}.",
              "At {g} goals in {gp} games, {player} projects to about {pace}.", "Project {player}'s {g} goals over a full 82 games and you get {pace}."],
        "r": ["Paces are paces, but that's a serious start.", "If he stays healthy, that's a career year.", "Write that one down and check back in March."],
    },
    "league_race": {
        "s": ["Let's look around the league for a second.", "Who's leading the scoring race right now?",
              "You keep an eye on the league leaders?", "There's a race shaping up at the top of the scoring list."],
        "p": ["{player} of {team} leads the league with {pts} points, {gap} ahead of {second}.", "Top of the list is {player}, {pts} points, and {second} is chasing, {gap} back.",
              "{player} has {pts} points to lead everybody. {second} is {gap} behind.", "It's {player} at {pts}. Closest pursuer is {second}, {gap} points back."],
        "r": ["Long way to go, but that's a fun race.", "That gap can disappear in one good week.", "Somebody's going to make a run at him."],
    },
    "point_streak": {
        "s": ["Somebody's running hot right now.", "Keep an eye on one player tonight.",
              "There's a streak alive on this ice.", "You want a stat for the scorer's table?", "One player just keeps finding the scoresheet."],
        "p": ["{player} has a point in {n} straight games.", "{player} comes in on a {n}-game point streak.",
              "{n} games in a row now with at least a point for {player}.", "{player}'s point streak is up to {n}."],
        "r": ["He's seeing the ice really well.", "Streaks like that are about confidence.", "Let's see if he extends it tonight."],
    },
    "slump": {
        "s": ["Not everything's going right for one scorer.", "There's a player here who'd love one to go in.",
              "Every scorer goes through it at some point.", "One name that's been fighting it lately."],
        "p": ["{player} hasn't scored in {n} games. He's at {g} on the year.", "{player}'s goal drought is at {n} games.",
              "{n} straight games without a goal for {player}.", "{player} has gone {n} games since his last goal."],
        "r": ["One ugly one off a skate and it all comes back.", "He's still getting looks, that's what matters.", "Shoot the puck, it'll turn."],
    },
    "goalie_form": {
        "s": ["Let's talk goaltending.", "How's the crease looking for {team}?",
              "The goalie has been a story this season.", "Here's the number for the man in the blue paint.", "Between the pipes for a moment."],
        "p": ["{goalie} is {w} and {l} with a {svp} save percentage.", "{goalie} has a {svp} save percentage and {w} wins.",
              "{goalie}: {w} wins, {l} losses, stopping {svp} of the shots he sees.", "{goalie} is sitting at a {svp} save percentage on the year."],
        "r": ["Goaltending is the whole game some nights.", "That's what keeps you in the hunt.", "He gives them a chance every night."],
    },
    "standings": {
        "s": ["Let's check the standings.", "Where does {team} sit right now?",
              "Big picture for a second.", "Before the next shift, the table."],
        "p": ["{team} is {pos} in the {division} with {pts} points from {gp} games.", "{team} sits {pos} in the {division}, {pts} points in {gp} games.",
              "{pts} points from {gp} games has {team} {pos} in the {division}.", "{team}: {pos} place in the {division}, {pts} points."],
        "r": ["Every point matters in that division.", "Plenty of hockey left, but that's where they are.", "It's tight in there."],
    },
    "race_in": {
        "s": ["Playoff picture time.", "How's the race looking for {team}?",
              "Let's look at the cut line.", "If the season ended today?"],
        "p": ["{team} would be in, {delta} points above the line.", "{team} has a cushion of {delta} points over the playoff cut.",
              "They're {delta} points clear of the line right now.", "{team} is in a playoff spot by {delta} points."],
        "r": ["No time to relax, though.", "That cushion can melt fast.", "Bank the points while you can."],
    },
    "race_out": {
        "s": ["Playoff picture time.", "How's the race looking for {team}?",
              "Let's look at the cut line.", "If the season ended today?"],
        "p": ["{team} would be on the outside, {delta} points back of a spot.", "{team} is {delta} points behind the last playoff spot.",
              "They're chasing, {delta} points off the pace.", "{delta} points is the gap {team} needs to close."],
        "r": ["Games like this are the ones that make it up.", "Plenty of road left to close that.", "They need a run."],
    },
    "last_meeting": {
        "s": ["These two have seen each other already this year.", "Remember the last time these teams met?",
              "There's some recent history here.", "Last meeting between these clubs?"],
        "p": ["{winner} won it {score} on day {day} of the season.", "It finished {score} for {winner} last time out.",
              "{winner} took that one, {score}.", "That was a {score} win for {winner}."],
        "r": ["Both benches will remember that.", "You can bet that came up in the video room.", "Different night, though."],
    },
    "season_series": {
        "s": ["How's the season series looking?", "These two keep running into each other.",
              "Head to head this year?", "Let's check the series."],
        "p": ["{team} has won {w} of the {n} meetings so far.", "It's {w} wins for {team} in {n} games between them.",
              "{team} leads the series {w} games to {l}.", "{team} has {w} of {n} this year."],
        "r": ["A win tonight changes the feel of it.", "Familiarity breeds a little edge.", "They know each other's habits by now."],
    },
    "rookie": {
        "s": ["Let's talk about the kid.", "There's a rookie on this ice worth watching.",
              "One of the young guys has been a revelation.", "You want a rookie to keep an eye on?"],
        "p": ["{player}, {age} years old, has {pts} points in {gp} games.", "{player} is {age} and already has {pts} points.",
              "The {age}-year-old {player} has {g} goals and {pts} points.", "{player}: {pts} points in his first {gp} games."],
        "r": ["Poise beyond his years.", "He plays like he's been here five seasons.", "That's a big piece of the future."],
    },
    "growth": {
        "s": ["Here's something you don't see in the box score.", "Development story for you.",
              "One player has taken a real step this year.", "The scouts have noticed something.", "Talk about a player who's grown up fast."],
        "p": ["{player} has gone from a {from_ovr} to a {to_ovr} overall since the start of the season.", "{player}'s rating is up {delta} since opening night, now at {to_ovr}.",
              "{player} started the year at {from_ovr}. He's a {to_ovr} now.", "Up {delta} points on the ratings this season, {player} is playing like a {to_ovr}."],
        "r": ["That's the coaching staff earning their money.", "That kind of jump changes a depth chart.", "Confidence is a hell of a drug for a hockey player."],
    },
    "decline": {
        "s": ["Not every rating goes up.", "There's one player who's been searching for his game.",
              "The numbers have been unkind to one veteran.", "Development works both ways."],
        "p": ["{player} has dipped from {from_ovr} to {to_ovr} since the start of the year.", "{player}'s rating is down {delta}, from {from_ovr} to {to_ovr}.",
              "{player} came in at {from_ovr} and he's at {to_ovr} now.", "A {delta}-point slide on the ratings for {player} this season."],
        "r": ["He'll work his way back.", "Sometimes it's just a matter of a role change.", "One good stretch fixes all of that."],
    },
    "outperform": {
        "s": ["Here's a player outperforming his billing.", "Not every star has the biggest rating.",
              "Somebody on this roster is punching above his weight.", "The ratings say one thing, the scoresheet says another."],
        "p": ["{player} is rated a {ovr}, but he's got {pts} points, {rank} on the team.", "{player}, a {ovr} overall, sits {rank} in team scoring with {pts}.",
              "At a {ovr} overall, {player} has no business having {pts} points. But he does.", "{pts} points from a {ovr}-rated {player}, good for {rank} on the club."],
        "r": ["Hockey sense doesn't show up on a rating sheet.", "Some players just find the soft ice.", "Give that man a raise."],
    },
    "contract_year": {
        "s": ["Somebody's playing for a contract.", "There's a business side to this season for one player.",
              "Contract talk for a second.", "One guy has a lot riding on this year."],
        "p": ["{player} is in the final year of his deal, and he has {pts} points.", "{player} has one year left on his contract and {pts} points so far.",
              "Last year of the deal for {player}. {pts} points and counting.", "{player} hits the market after this season. He's at {pts} points."],
        "r": ["Nothing motivates like a contract year.", "His agent is watching this one.", "That's a nice negotiating position."],
    },
    "injury_return": {
        "s": ["Good to see one player back out there.", "A welcome return tonight.",
              "One player is fresh off the injury list.", "Health update for you."],
        "p": ["{player} missed {games} games and is back in the lineup.", "{player} is back after a {games}-game absence.",
              "After sitting out {games} games, {player} returns.", "{player} is healthy again after missing {games}."],
        "r": ["Nobody likes watching from the press box.", "It takes a few shifts to get your legs back.", "Big boost for the lineup."],
    },
    "coach_record": {
        "s": ["Let's talk about the bench boss.", "How's the coach's season going?",
              "The man behind the {team} bench.", "Coaching check-in."],
        "p": ["{coach} has {team} at {record} this season.", "{team} is {record} under {coach}.",
              "{coach}'s record this year: {record}.", "Under {coach}, {team} has gone {record}."],
        "r": ["Every coach lives and dies by that record.", "He's got them playing a clear system.", "The room seems to believe in him."],
    },
    "headline": {
        "s": ["Did you see the story going around this week?", "One of the headlines out of {team} camp.",
              "Talk around the league this week.", "Here's what people are talking about."],
        "p": ["{headline}.", "The word is: {headline}.", "Headline reads: {headline}.", "Simply put: {headline}."],
        "r": ["That'll be the talk of the dressing room.", "Every team has its storylines.", "We'll see how it plays out."],
    },
    "team_form": {
        "s": ["How's {team} been playing lately?", "Recent form for {team}.",
              "Let's look at the last ten.", "Momentum check."],
        "p": ["{team} is {last10} over its last ten games.", "Last ten for {team}: {last10}.",
              "{team} has gone {last10} in its last ten.", "A {last10} stretch over ten games for {team}."],
        "r": ["That tells you plenty about the mood.", "Form matters in this league.", "Streaks come and go, but that's real."],
    },
    "scoring_rates": {
        "s": ["Goals for, goals against.", "How's the goal differential for {team}?",
              "Let's do the math on {team}.", "Offence and defence in one number."],
        "p": ["{team} is scoring {gfpg} a game and allowing {gapg}.", "{team} averages {gfpg} goals for, {gapg} against.",
              "{gfpg} goals per game for {team}, {gapg} allowed.", "{team}'s goal difference sits at {diff}."],
        "r": ["That's the number that wins you games.", "Keep that gap and you're in good shape.", "It all comes back to keeping it out of your net."],
    },
    "home_road": {
        "s": ["Home and road splits, quickly.", "Does {team} travel well?",
              "There's a split worth mentioning.", "Building versus road."],
        "p": ["{team} is {home} at home and {away} on the road.", "Home record {home}, road record {away} for {team}.",
              "{team}: {home} in its own building, {away} away.", "On the road {team} is {away}. At home, {home}."],
        "r": ["Last change matters more than people think.", "A good road team is a dangerous team.", "Comfort of your own bed."],
    },
    "pp_scorer": {
        "s": ["Watch for this player when the power play comes.", "One name on the man advantage.",
              "Special teams note.", "Who's the power-play weapon?"],
        "p": ["{player} has {ppg} power-play goals already.", "{ppg} of {player}'s goals have come on the power play.",
              "{player} leads the way with {ppg} power-play goals.", "On the man advantage it's {player}, with {ppg} goals."],
        "r": ["Give him room on the half-wall at your peril.", "You know where that puck is going.", "That's a weapon."],
    },
    "physical": {
        "s": ["Somebody's been throwing his weight around.", "The physical side of things.",
              "Not every contribution is a point.", "Let's give the grinders some love."],
        "p": ["{player} leads {team} with {hits} hits.", "{hits} hits on the season for {player}.",
              "{player} has delivered {hits} hits this year.", "Top hitter on the club is {player}, {hits} of them."],
        "r": ["Every team needs that guy.", "Defencemen hear him coming.", "That wears you down over sixty minutes."],
    },
    "blocker": {
        "s": ["One player sacrifices the body more than anyone.", "Shot-blocking note.",
              "Unsung work in front of the net.", "This is the stuff coaches love."],
        "p": ["{player} has {blk} blocked shots this season.", "{blk} blocks for {player} so far.",
              "{player} leads the club with {blk} blocked shots.", "Nobody on {team} blocks more than {player}. {blk} of them."],
        "r": ["That takes courage.", "Ice packs after every game for that guy.", "Coaches love that commitment."],
    },
    "shots_tonight": {
        "s": ["Look at the shot clock tonight.", "Tonight's shot totals tell a story.",
              "Who's carrying the play so far?", "Quick look at the shots."],
        "p": ["{team} has outshot {opp} {a} to {b}.", "Shots on goal: {team} {a}, {opp} {b}.",
              "{team} leads {opp} on the shot clock, {a} to {b}.", "It's {a} shots to {b} in favour of {team}."],
        "r": ["Volume doesn't always win, but it helps.", "That's pressure that eventually pays.", "The goalie at the other end has been busy."],
    },
    "on_ice_line": {
        "s": ["He's on the ice right now, so let's talk about him.", "Look who's out there now.",
              "There's a guy on this shift worth a mention.", "Watch number one on this shift."],
        "p": ["{player} has {g} goals and {a} assists this season.", "{player}: {pts} points in {gp} games.",
              "{player} comes in with {g} goals, {pts} points.", "That's {player}, {pts} points so far this year."],
        "r": ["Always dangerous when he's out there.", "You feel it when he touches the puck.", "Steady contributor."],
    },
    "rivalry_record": {
        "s": ["These two have some history in this save.", "Rivalry check.",
              "The bad blood between these clubs.", "Head to head over the years."],
        "p": ["{team} and {opp} have played {gp} times, with {n_one} one-goal games.", "{gp} meetings, {n_one} of them decided by a single goal.",
              "Their history: {gp} games, {series} playoff series.", "It's {gp} games between them and {n_one} one-goal finishes."],
        "r": ["That's why this one has an edge.", "Nobody likes losing to them.", "Rivalries make the season."],
    },
    "past_champion": {
        "s": ["A little history from this league.", "Remember who won it all last time?",
              "Champions talk.", "Look back at last season for a second."],
        "p": ["{champion} won the championship in {season}.", "The {season} title went to {champion}.",
              "{champion} were champions in {season}.", "{season} belonged to {champion}."],
        "r": ["Everyone's chasing them now.", "Defending a title is the hardest thing in sport.", "That banner hangs over every game."],
    },
    # ── stable real-world history
    "hist_cups": {
        "s": ["A little franchise history for you.", "Some history on this club.",
              "Before the next faceoff, a history note.", "Tradition matters in this building."],
        "p": ["{team} has won the Stanley Cup {cups} times.", "{cups} Stanley Cups in {team} history.",
              "That franchise has {cups} championships to its name.", "{team} has lifted the Cup {cups} times."],
        "r": ["The banners tell the story.", "That's a lot of history to live up to.", "Fans here expect a lot."],
    },
    "hist_drought": {
        "s": ["Here's one that stings for the locals.", "A history note the fans know too well.",
              "Still chasing the first one.", "Some franchise trivia."],
        "p": ["{team} has been in the league since {first_season} and is still looking for its first Cup.", "No Stanley Cup yet for {team}, which joined in {first_season}.",
              "{team} entered the league in {first_season}. The Cup hasn't come home yet.", "Since {first_season}, {team} has waited for its first title."],
        "r": ["One day it'll happen, and this place will explode.", "That's what keeps them coming back.", "Every season starts with that hope."],
    },
    "hist_reloc": {
        "s": ["Trivia question for you.", "Here's one for the history buffs.",
              "Not everyone remembers this.", "A little franchise backstory."],
        "p": ["This franchise started out in {prev_city} back in {first_season}.", "Before {team}, this franchise played in {prev_city}, starting in {first_season}.",
              "The club began life in {prev_city} in {first_season}.", "Its first home was {prev_city}, {first_season}."],
        "r": ["Funny how these things work out.", "A lot of history on the move.", "Different city, same jersey on the hearts now."],
    },
    "hist_age": {
        "s": ["How long has this franchise been around?", "A quick history note.",
              "Some perspective on this club.", "History for you between whistles."],
        "p": ["{team} has been in the league since {first_season}.", "The franchise dates back to {first_season}.",
              "{first_season} was the club's first season.", "{team} joined the league in {first_season}."],
        "r": ["A lot of seasons, a lot of stories.", "Generations of fans in those seats.", "Time flies in this game."],
    },
    # ── generic booth colour (always eligible; city slot from the home team)
    "atmosphere": {
        "s": ["Listen to this crowd.", "The building has a buzz tonight.",
              "You can feel it in here.", "Good energy in {city} tonight."],
        "p": ["This crowd knows its hockey. They react to the little plays.", "You can hear the sticks on the ice from up here when it goes quiet.",
              "They're on their feet for a dump-in. That's a hockey town.", "Every hit gets a roar. They want a physical game."],
        "r": ["I love working in buildings like this.", "That's why you play the game.", "Great place to call a game."],
    },
    "road_trip": {
        "s": ["Life on the road for a minute.", "You know what the schedule's been like.",
              "Let's talk about travel.", "The grind of the season."],
        "p": ["Early flights, late arrivals, and you still have to bring legs every night.", "Hotel, rink, bus, plane. Players will tell you it's the travel that wears you down.",
              "A back-to-back after a late flight is the toughest test in this sport.", "By this point of the year, everybody's playing through something."],
        "r": ["And we just sit up here and judge them.", "That's why depth matters so much.", "The trainers earn their money this time of year."],
    },
    "culture": {
        "s": ["You know what I love about this sport?", "Little thing I've always appreciated.",
              "There's a code in hockey.", "Something you don't see on TV."],
        "p": ["Guys block shots with their bodies and then tap the goalie's pads like it's nothing.", "Teammates stand up for each other. Every single night.",
              "The handshake line after a playoff series. Nothing else like it in sport.", "The stick taps from the bench after a good shift. Respect, no words needed."],
        "r": ["That's the soul of the game right there.", "Never gets old.", "You don't teach that, it's in them."],
    },
    "booth": {
        "s": ["I have to tell you, the coffee up here tonight...", "My producer is waving at me, so let me ask you something.",
              "Can we talk about the press box food?", "You brought your lucky pen tonight?"],
        "p": ["Strong enough to keep me sharp through overtime, let's put it that way.", "He wants us to talk about the game. Revolutionary concept.",
              "Somebody ordered the soup again. Brave choice.", "I've had the same pen for six seasons. Not changing now."],
        "r": ["Back to hockey, before we get fired.", "Okay, okay, eyes on the ice.", "Let's get back to it."],
    },
}

EN_CUTIN = ["Hold that thought.", "Oh, here we go.", "Wait, wait.", "Hang on a second.", "Stop right there."]
EN_RESUME = ["Anyway, back to {topic}.", "Where was I? Right, {topic}.", "As I was saying about {topic}."]


def _mk(lang_tables: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    return lang_tables


# ════════════════════════════════════════════════════════════════════ FRENCH
FR: Dict[str, Dict[str, List[str]]] = {
    "team_leader": {"s": ["Qui porte l'attaque de {team} cette saison?", "Parlons de l'offensive de {team}.", "Un nom à retenir chez {team}?"],
                    "p": ["C'est {player}, avec {pts} points en {gp} matchs.", "{player} mène l'équipe avec {pts} points, dont {g} buts.", "{player} : {g} buts, {a} passes, {pts} points."],
                    "r": ["Toute l'attaque passe par lui.", "Difficile de discuter avec ces chiffres."]},
    "goal_pace": {"s": ["Un chiffre m'a frappé aujourd'hui.", "J'ai fait un petit calcul à l'hôtel."],
                  "p": ["{player} a {g} buts en {gp} matchs, un rythme de {pace} buts sur une saison complète.", "Au rythme actuel, {player} finirait avec environ {pace} buts.", "Avec {g} buts en {gp} matchs, {player} vise les {pace}."],
                  "r": ["Ce n'est qu'une projection, mais quel début.", "S'il reste en santé, ce sera sa meilleure saison."]},
    "league_race": {"s": ["Faisons le tour de la ligue.", "Qui mène la course aux points?", "La course au sommet se dessine."],
                    "p": ["{player}, de {team}, mène avec {pts} points, {gap} devant {second}.", "{player} est en tête avec {pts} points. {second} suit à {gap} points.", "C'est {player} à {pts} points, et {second} le talonne."],
                    "r": ["Il reste beaucoup de hockey.", "Cet écart peut fondre en une semaine."]},
    "point_streak": {"s": ["Quelqu'un est en feu en ce moment.", "Surveillez un joueur ce soir.", "Une séquence est toujours vivante."],
                     "p": ["{player} a obtenu un point dans {n} matchs consécutifs.", "{player} arrive avec une séquence de {n} matchs avec au moins un point.", "{n} matchs de suite avec un point pour {player}."],
                     "r": ["La confiance, ça change tout.", "Voyons s'il la prolonge ce soir."]},
    "goalie_form": {"s": ["Parlons des gardiens.", "Comment ça va devant le filet pour {team}?", "Le gardien est une histoire en soi."],
                    "p": ["{goalie} présente une fiche de {w} victoires et {l} défaites, avec un taux d'arrêts de {svp}.", "{goalie} affiche un taux d'arrêts de {svp}.", "{goalie} : {w} victoires et un taux de {svp}."],
                    "r": ["Le gardien, c'est la moitié de l'équipe.", "Il donne une chance à son équipe chaque soir."]},
    "standings": {"s": ["Jetons un œil au classement.", "Où se situe {team}?", "Le portrait global."],
                  "p": ["{team} occupe le rang {pos_n} de la section {division} avec {pts} points en {gp} matchs.", "{team} est au rang {pos_n} de sa section, {pts} points.", "{pts} points en {gp} matchs pour {team}, rang {pos_n} dans la section {division}."],
                  "r": ["Chaque point compte.", "C'est serré là-dedans."]},
    "rookie": {"s": ["Parlons du jeune.", "Une recrue à surveiller ce soir.", "Un des jeunes est une belle surprise."],
               "p": ["{player}, {age} ans, a déjà {pts} points en {gp} matchs.", "À {age} ans, {player} compte {g} buts et {pts} points.", "{player} : {pts} points à ses {gp} premiers matchs."],
               "r": ["Il joue avec un calme étonnant.", "C'est l'avenir de l'équipe."]},
    "growth": {"s": ["Voici une histoire de développement.", "Un joueur a franchi une étape cette saison.", "Les dépisteurs ont remarqué quelque chose."],
               "p": ["{player} est passé de {from_ovr} à {to_ovr} au classement global depuis le début de la saison.", "La cote de {player} a grimpé de {delta}, à {to_ovr}.", "{player} a commencé l'année à {from_ovr}. Il est à {to_ovr} aujourd'hui."],
               "r": ["Le personnel d'entraîneurs fait du beau travail.", "Ce genre de progression change un alignement."]},
    "coach_record": {"s": ["Parlons de l'entraîneur.", "Derrière le banc de {team}."],
                     "p": ["{coach} mène {team} à une fiche de {record}.", "Sous {coach}, {team} présente une fiche de {record}.", "La fiche de {coach} cette saison : {record}."],
                     "r": ["Le vestiaire croit en lui.", "Son système est clair."]},
    "team_form": {"s": ["Comment va {team} ces temps-ci?", "La forme récente de {team}."],
                  "p": ["{team} présente une fiche de {last10} à ses dix derniers matchs.", "Dix derniers matchs pour {team} : {last10}.", "{team} a joué pour {last10} sur dix matchs."],
                  "r": ["Ça en dit long sur l'ambiance.", "La forme compte dans cette ligue."]},
    "shots_tonight": {"s": ["Regardez les tirs ce soir.", "Qui contrôle le jeu jusqu'ici?"],
                      "p": ["{team} domine {opp} {a} à {b} au chapitre des tirs.", "Tirs au but : {team} {a}, {opp} {b}.", "C'est {a} tirs contre {b} en faveur de {team}."],
                      "r": ["La pression finit par payer.", "Le gardien d'en face travaille fort."]},
    "headline": {"s": ["Avez-vous vu la nouvelle cette semaine?", "On en parle dans la ligue."],
                 "p": ["{headline}.", "La manchette : {headline}.", "En bref : {headline}."],
                 "r": ["Ça va faire jaser dans le vestiaire.", "On verra comment ça tourne."]},
    "hist_cups": {"s": ["Un peu d'histoire.", "Une note historique sur cette équipe."],
                  "p": ["{team} a remporté la coupe Stanley {cups} fois.", "{cups} coupes Stanley dans l'histoire de {team}.", "Cette concession compte {cups} championnats."],
                  "r": ["Les bannières racontent l'histoire.", "Les partisans en veulent toujours plus."]},
    "hist_drought": {"s": ["Une note d'histoire qui fait mal ici.", "Toujours à la recherche de la première."],
                     "p": ["{team} est dans la ligue depuis {first_season} et attend toujours sa première coupe.", "Pas encore de coupe Stanley pour {team}, arrivée en {first_season}.", "Depuis {first_season}, {team} rêve de son premier titre."],
                     "r": ["Le jour où ça arrive, l'aréna va exploser.", "C'est ce qui ramène les partisans."]},
    "atmosphere": {"s": ["Écoutez cette foule.", "Il y a de l'ambiance à {city} ce soir."],
                   "p": ["Cette foule connaît son hockey, elle réagit aux petits jeux.", "Chaque mise en échec soulève la foule.", "Ils se lèvent pour un dégagement. C'est une ville de hockey."],
                   "r": ["J'adore travailler dans ces amphithéâtres.", "C'est pour ça qu'on joue."]},
    "road_trip": {"s": ["Parlons des voyages.", "La routine d'une saison."],
                  "p": ["Vols tôt le matin, arrivées tard, et il faut quand même avoir des jambes.", "Hôtel, aréna, autobus, avion. C'est le voyage qui use les joueurs.", "Un deuxième match en deux soirs après un vol tardif, c'est le test ultime."],
                  "r": ["Et nous, on juge d'ici.", "D'où l'importance de la profondeur."]},
    "booth": {"s": ["Je dois vous dire, le café ce soir...", "Mon réalisateur me fait des signes."],
              "p": ["Assez fort pour me garder éveillé jusqu'en prolongation.", "Il veut qu'on parle du match. Quelle idée.", "Quelqu'un a encore commandé la soupe. Courageux."],
              "r": ["Revenons au hockey.", "D'accord, les yeux sur la glace."]},
}
FR_CUTIN = ["Attendez une seconde.", "Oh, ça part.", "Attention, attention.", "Gardez ça en tête."]
FR_RESUME = ["Bref, revenons à {topic}.", "Je disais donc, {topic}."]

# ════════════════════════════════════════════════════════════════════ SPANISH
ES: Dict[str, Dict[str, List[str]]] = {
    "team_leader": {"s": ["¿Quién lleva el ataque de {team} esta temporada?", "Hablemos de la ofensiva de {team}.", "¿Un nombre en la plantilla de {team}?"],
                    "p": ["Es {player}, con {pts} puntos en {gp} partidos.", "{player} lidera al equipo con {pts} puntos, {g} de ellos goles.", "{player}: {g} goles, {a} asistencias, {pts} puntos."],
                    "r": ["Todo el ataque pasa por él.", "Esos números no se discuten."]},
    "goal_pace": {"s": ["Hoy me llamó la atención un dato.", "Estuve haciendo cuentas en el hotel."],
                  "p": ["{player} lleva {g} goles en {gp} partidos, un ritmo de {pace} goles en una temporada completa.", "A este ritmo, {player} terminaría con unos {pace} goles.", "Con {g} goles en {gp} partidos, {player} apunta a {pace}."],
                  "r": ["Es solo una proyección, pero qué comienzo.", "Si sigue sano, será su mejor año."]},
    "league_race": {"s": ["Demos una vuelta por la liga.", "¿Quién lidera la carrera de puntos?", "Se perfila una carrera en la cima."],
                    "p": ["{player}, de {team}, lidera con {pts} puntos, {gap} por delante de {second}.", "{player} va primero con {pts} puntos. {second} le sigue a {gap}.", "Es {player} con {pts} puntos, y {second} lo persigue."],
                    "r": ["Queda mucho hockey.", "Esa ventaja puede desaparecer en una semana."]},
    "point_streak": {"s": ["Alguien está en racha.", "Atentos a un jugador esta noche.", "Hay una racha viva en esta pista."],
                     "p": ["{player} ha sumado al menos un punto en {n} partidos seguidos.", "{player} llega con una racha de {n} partidos con punto.", "{n} partidos seguidos con punto para {player}."],
                     "r": ["La confianza lo cambia todo.", "Veamos si la alarga esta noche."]},
    "goalie_form": {"s": ["Hablemos de porteros.", "¿Cómo está la portería de {team}?", "El portero ha sido toda una historia."],
                    "p": ["{goalie} tiene {w} victorias y {l} derrotas, con un porcentaje de paradas de {svp}.", "{goalie} para el {svp} de los tiros que recibe.", "{goalie}: {w} victorias y un {svp} de paradas."],
                    "r": ["El portero es medio equipo.", "Les da una oportunidad cada noche."]},
    "standings": {"s": ["Miremos la clasificación.", "¿Dónde está {team}?", "La foto general."],
                  "p": ["{team} ocupa el puesto {pos_n} de la división {division} con {pts} puntos en {gp} partidos.", "{team} está en el puesto {pos_n} de su división, {pts} puntos.", "{pts} puntos en {gp} partidos para {team}, puesto {pos_n} en la {division}."],
                  "r": ["Cada punto cuenta.", "Está muy apretado."]},
    "rookie": {"s": ["Hablemos del novato.", "Un novato a seguir esta noche.", "Uno de los jóvenes ha sido una revelación."],
               "p": ["{player}, con {age} años, ya suma {pts} puntos en {gp} partidos.", "Con {age} años, {player} lleva {g} goles y {pts} puntos.", "{player}: {pts} puntos en sus primeros {gp} partidos."],
               "r": ["Juega con una calma impropia de su edad.", "Es el futuro del equipo."]},
    "growth": {"s": ["Una historia de desarrollo.", "Un jugador ha dado un salto esta temporada.", "Los ojeadores han notado algo."],
               "p": ["{player} ha pasado de {from_ovr} a {to_ovr} de valoración desde el inicio de la temporada.", "La valoración de {player} ha subido {delta}, hasta {to_ovr}.", "{player} empezó el año en {from_ovr}. Ahora es un {to_ovr}."],
               "r": ["El cuerpo técnico está haciendo un gran trabajo.", "Un salto así cambia la rotación."]},
    "coach_record": {"s": ["Hablemos del entrenador.", "El hombre detrás del banquillo de {team}."],
                     "p": ["{coach} tiene a {team} con un récord de {record}.", "Con {coach}, {team} va {record}.", "El balance de {coach} esta temporada: {record}."],
                     "r": ["El vestuario cree en él.", "Su sistema está claro."]},
    "team_form": {"s": ["¿Cómo llega {team} últimamente?", "La forma reciente de {team}."],
                  "p": ["{team} va {last10} en sus últimos diez partidos.", "Últimos diez de {team}: {last10}.", "{team} lleva {last10} en diez partidos."],
                  "r": ["Eso dice mucho del ánimo.", "La forma importa en esta liga."]},
    "shots_tonight": {"s": ["Miren los tiros esta noche.", "¿Quién manda en el juego?"],
                      "p": ["{team} supera a {opp} en tiros, {a} a {b}.", "Tiros a puerta: {team} {a}, {opp} {b}.", "Son {a} tiros contra {b} a favor de {team}."],
                      "r": ["La presión acaba dando frutos.", "El portero rival está trabajando mucho."]},
    "headline": {"s": ["¿Vieron la noticia de esta semana?", "Se comenta en toda la liga."],
                 "p": ["{headline}.", "El titular: {headline}.", "En pocas palabras: {headline}."],
                 "r": ["Será tema en el vestuario.", "Veremos cómo termina."]},
    "hist_cups": {"s": ["Un poco de historia.", "Una nota histórica de este equipo."],
                  "p": ["{team} ha ganado la Copa Stanley {cups} veces.", "{cups} Copas Stanley en la historia de {team}.", "Esta franquicia tiene {cups} campeonatos."],
                  "r": ["Los estandartes cuentan la historia.", "Aquí la afición exige mucho."]},
    "hist_drought": {"s": ["Una nota histórica que duele aquí.", "Todavía buscan la primera."],
                     "p": ["{team} está en la liga desde {first_season} y sigue buscando su primera Copa.", "Todavía no hay Copa Stanley para {team}, que llegó en {first_season}.", "Desde {first_season}, {team} sueña con su primer título."],
                     "r": ["El día que llegue, esto va a explotar.", "Esa ilusión trae a la gente cada noche."]},
    "atmosphere": {"s": ["Escuchen a esta afición.", "Hay ambiente en {city} esta noche."],
                   "p": ["Esta afición sabe de hockey, reacciona a los pequeños detalles.", "Cada choque levanta al público.", "Se ponen de pie por un despeje. Es una ciudad de hockey."],
                   "r": ["Me encanta narrar en estos pabellones.", "Por esto se juega."]},
    "road_trip": {"s": ["Hablemos de los viajes.", "El desgaste de la temporada."],
                  "p": ["Vuelos temprano, llegadas tarde, y aun así hay que tener piernas.", "Hotel, pista, autobús, avión. Los viajes son lo que desgasta.", "Un segundo partido seguido tras un vuelo tardío es la prueba más dura."],
                  "r": ["Y nosotros opinando desde aquí.", "Por eso importa la profundidad."]},
    "booth": {"s": ["Tengo que decirles, el café de esta noche...", "Mi productor me está haciendo señas."],
              "p": ["Lo bastante fuerte para aguantar hasta la prórroga.", "Quiere que hablemos del partido. Qué ocurrencia.", "Alguien volvió a pedir la sopa. Valiente."],
              "r": ["Volvamos al hockey.", "Vale, los ojos en la pista."]},
}
ES_CUTIN = ["Un momento.", "Atención, que viene.", "Espera, espera.", "Guarda esa idea."]
ES_RESUME = ["En fin, volvamos a {topic}.", "Como decía, {topic}."]

# ════════════════════════════════════════════════════════════════════ GERMAN
DE: Dict[str, Dict[str, List[str]]] = {
    "team_leader": {"s": ["Wer trägt die Offensive von {team} in dieser Saison?", "Reden wir über den Angriff von {team}.", "Ein Name im Kader von {team}?"],
                    "p": ["Das ist {player}, mit {pts} Punkten aus {gp} Spielen.", "{player} führt das Team mit {pts} Punkten an, {g} davon Tore.", "{player}: {g} Tore, {a} Vorlagen, {pts} Punkte."],
                    "r": ["Alles läuft über ihn.", "Diese Zahlen sprechen für sich."]},
    "goal_pace": {"s": ["Eine Zahl ist mir heute aufgefallen.", "Ich habe im Hotel ein bisschen gerechnet."],
                  "p": ["{player} hat {g} Tore in {gp} Spielen, das sind hochgerechnet {pace} Tore auf eine volle Saison.", "Bei diesem Tempo käme {player} auf etwa {pace} Tore.", "Mit {g} Toren aus {gp} Spielen steuert {player} auf {pace} zu."],
                  "r": ["Nur eine Hochrechnung, aber was für ein Start.", "Wenn er gesund bleibt, wird das sein bestes Jahr."]},
    "league_race": {"s": ["Ein Blick in die Liga.", "Wer führt die Scorerliste an?", "Oben bahnt sich ein Rennen an."],
                    "p": ["{player} von {team} führt mit {pts} Punkten, {gap} vor {second}.", "{player} liegt mit {pts} Punkten vorn. {second} folgt mit {gap} Punkten Rückstand.", "{player} mit {pts} Punkten, {second} ist ihm auf den Fersen."],
                    "r": ["Es ist noch viel Eishockey zu spielen.", "Der Vorsprung kann in einer Woche weg sein."]},
    "point_streak": {"s": ["Da ist einer gerade heiß.", "Achten Sie heute auf einen Spieler.", "Hier läuft eine Serie."],
                     "p": ["{player} hat in {n} Spielen in Folge gepunktet.", "{player} kommt mit einer Punkteserie von {n} Spielen.", "{n} Spiele in Folge mit Scorerpunkt für {player}."],
                     "r": ["Selbstvertrauen ist alles.", "Mal sehen, ob er sie heute ausbaut."]},
    "goalie_form": {"s": ["Reden wir über die Torhüter.", "Wie steht es um das Tor von {team}?", "Der Torhüter ist eine Geschichte für sich."],
                    "p": ["{goalie} steht bei {w} Siegen und {l} Niederlagen, mit einer Fangquote von {svp}.", "{goalie} hält {svp} der Schüsse auf sein Tor.", "{goalie}: {w} Siege und eine Fangquote von {svp}."],
                    "r": ["Der Torhüter ist die halbe Mannschaft.", "Er gibt seinem Team jeden Abend eine Chance."]},
    "standings": {"s": ["Ein Blick auf die Tabelle.", "Wo steht {team}?", "Das große Bild."],
                  "p": ["{team} steht auf Platz {pos_n} der {division} Division mit {pts} Punkten aus {gp} Spielen.", "{team} belegt Rang {pos_n} in der Division, {pts} Punkte.", "{pts} Punkte aus {gp} Spielen für {team}, Platz {pos_n}."],
                  "r": ["Jeder Punkt zählt.", "Da geht es eng zu."]},
    "rookie": {"s": ["Reden wir über den Rookie.", "Ein Neuling, den man heute beobachten sollte.", "Einer der Jungen ist eine echte Entdeckung."],
               "p": ["{player}, {age} Jahre alt, hat schon {pts} Punkte in {gp} Spielen.", "Mit {age} Jahren hat {player} {g} Tore und {pts} Punkte.", "{player}: {pts} Punkte in seinen ersten {gp} Spielen."],
               "r": ["Er spielt mit einer Ruhe, die man in dem Alter selten sieht.", "Das ist die Zukunft des Teams."]},
    "growth": {"s": ["Eine Geschichte über Entwicklung.", "Ein Spieler hat in dieser Saison einen Schritt gemacht.", "Den Scouts ist etwas aufgefallen."],
               "p": ["{player} hat sich seit Saisonbeginn von {from_ovr} auf {to_ovr} Gesamtwertung gesteigert.", "Die Wertung von {player} ist um {delta} gestiegen, auf {to_ovr}.", "{player} begann die Saison bei {from_ovr}. Jetzt ist er eine {to_ovr}."],
               "r": ["Das Trainerteam leistet gute Arbeit.", "So ein Sprung verändert die Aufstellung."]},
    "coach_record": {"s": ["Reden wir über den Trainer.", "Der Mann hinter der Bank von {team}."],
                     "p": ["{coach} hat {team} auf eine Bilanz von {record} geführt.", "Unter {coach} steht {team} bei {record}.", "Die Bilanz von {coach} in dieser Saison: {record}."],
                     "r": ["Die Kabine glaubt an ihn.", "Sein System ist klar."]},
    "team_form": {"s": ["Wie läuft es zuletzt für {team}?", "Die aktuelle Form von {team}."],
                  "p": ["{team} hat aus den letzten zehn Spielen eine Bilanz von {last10}.", "Letzte zehn Spiele für {team}: {last10}.", "{team} steht bei {last10} in den letzten zehn."],
                  "r": ["Das sagt viel über die Stimmung.", "Form zählt in dieser Liga."]},
    "shots_tonight": {"s": ["Schauen Sie auf die Schüsse heute.", "Wer bestimmt das Spiel bisher?"],
                      "p": ["{team} hat {opp} mit {a} zu {b} Schüssen überboten.", "Torschüsse: {team} {a}, {opp} {b}.", "{a} zu {b} Schüsse für {team}."],
                      "r": ["Druck zahlt sich irgendwann aus.", "Der Torhüter drüben hat viel zu tun."]},
    "headline": {"s": ["Haben Sie die Meldung diese Woche gesehen?", "Darüber redet die Liga."],
                 "p": ["{headline}.", "Die Schlagzeile: {headline}.", "Kurz gesagt: {headline}."],
                 "r": ["Das wird in der Kabine Thema sein.", "Mal sehen, wie das ausgeht."]},
    "hist_cups": {"s": ["Ein bisschen Geschichte.", "Eine historische Notiz zu diesem Klub."],
                  "p": ["{team} hat den Stanley Cup {cups} Mal gewonnen.", "{cups} Stanley Cups in der Geschichte von {team}.", "Diese Franchise hat {cups} Meisterschaften."],
                  "r": ["Die Banner erzählen die Geschichte.", "Die Fans hier erwarten viel."]},
    "hist_drought": {"s": ["Eine Notiz, die hier wehtut.", "Immer noch auf der Jagd nach dem ersten Titel."],
                     "p": ["{team} spielt seit {first_season} in der Liga und wartet immer noch auf den ersten Cup.", "Noch kein Stanley Cup für {team}, seit {first_season} dabei.", "Seit {first_season} träumt {team} vom ersten Titel."],
                     "r": ["Wenn es soweit ist, explodiert diese Halle.", "Diese Hoffnung bringt die Leute jeden Abend her."]},
    "atmosphere": {"s": ["Hören Sie sich dieses Publikum an.", "Gute Stimmung heute in {city}."],
                   "p": ["Dieses Publikum versteht Eishockey, es reagiert auf die kleinen Dinge.", "Jeder Check wird bejubelt.", "Sie stehen für einen Befreiungsschlag auf. Das ist eine Eishockeystadt."],
                   "r": ["Ich liebe solche Hallen.", "Dafür spielt man."]},
    "road_trip": {"s": ["Reden wir über das Reisen.", "Der Alltag einer Saison."],
                  "p": ["Frühe Flüge, späte Ankünfte, und trotzdem müssen die Beine da sein.", "Hotel, Halle, Bus, Flugzeug. Das Reisen zehrt an den Spielern.", "Ein zweites Spiel in zwei Tagen nach einem späten Flug ist die härteste Prüfung."],
                  "r": ["Und wir urteilen von hier oben.", "Deshalb ist Tiefe so wichtig."]},
    "booth": {"s": ["Ich muss sagen, der Kaffee heute...", "Mein Produzent winkt mir zu."],
              "p": ["Stark genug für eine Verlängerung, sagen wir es so.", "Er will, dass wir über das Spiel reden. Was für eine Idee.", "Jemand hat wieder die Suppe bestellt. Mutig."],
              "r": ["Zurück zum Eishockey.", "Okay, Augen aufs Eis."]},
}
DE_CUTIN = ["Moment mal.", "Oh, jetzt geht's los.", "Warte, warte.", "Halt den Gedanken fest."]
DE_RESUME = ["Zurück zu {topic}.", "Wo war ich? Richtig, {topic}."]

# ════════════════════════════════════════════════════════════════════ SWEDISH
SV: Dict[str, Dict[str, List[str]]] = {
    "team_leader": {"s": ["Vem bär offensiven för {team} i år?", "Vi pratar om anfallsspelet i {team}.", "Ett namn i {team} att hålla koll på?"],
                    "p": ["Det är {player}, med {pts} poäng på {gp} matcher.", "{player} leder laget med {pts} poäng, varav {g} mål.", "{player}: {g} mål, {a} assist, {pts} poäng."],
                    "r": ["Allt går genom honom.", "Siffrorna talar för sig själva."]},
    "goal_pace": {"s": ["En siffra fångade min blick i dag.", "Jag räknade lite på hotellet."],
                  "p": ["{player} har {g} mål på {gp} matcher, det är takt för {pace} mål över en hel säsong.", "I den här takten landar {player} på runt {pace} mål.", "Med {g} mål på {gp} matcher siktar {player} mot {pace}."],
                  "r": ["Bara en prognos, men vilken start.", "Håller han sig frisk blir det hans bästa år."]},
    "league_race": {"s": ["Vi tittar runt i ligan.", "Vem leder poängligan?", "Det är ett race i toppen."],
                    "p": ["{player} i {team} leder med {pts} poäng, {gap} före {second}.", "{player} toppar med {pts} poäng. {second} ligger {gap} bakom.", "Det är {player} på {pts} poäng, och {second} jagar."],
                    "r": ["Mycket hockey kvar.", "Den ledningen kan försvinna på en vecka."]},
    "point_streak": {"s": ["Någon är het just nu.", "Håll ögonen på en spelare i kväll.", "Det finns en svit vid liv här."],
                     "p": ["{player} har gjort poäng i {n} raka matcher.", "{player} kommer in med en poängsvit på {n} matcher.", "{n} raka matcher med poäng för {player}."],
                     "r": ["Självförtroende är allt.", "Får se om han förlänger den i kväll."]},
    "goalie_form": {"s": ["Vi pratar målvakter.", "Hur ser det ut i målet för {team}?", "Målvakten har varit en historia i sig."],
                    "p": ["{goalie} har {w} vinster och {l} förluster, med en räddningsprocent på {svp}.", "{goalie} räddar {svp} av skotten han möter.", "{goalie}: {w} vinster och {svp} i räddningsprocent."],
                    "r": ["Målvakten är halva laget.", "Han ger laget en chans varje kväll."]},
    "standings": {"s": ["Vi kollar tabellen.", "Var ligger {team}?", "Den stora bilden."],
                  "p": ["{team} ligger på plats {pos_n} i {division} med {pts} poäng på {gp} matcher.", "{team} är på plats {pos_n} i divisionen, {pts} poäng.", "{pts} poäng på {gp} matcher för {team}, plats {pos_n}."],
                  "r": ["Varje poäng räknas.", "Det är tajt där inne."]},
    "rookie": {"s": ["Vi pratar om rookien.", "En nykomling att hålla koll på i kväll.", "En av de unga har varit en uppenbarelse."],
               "p": ["{player}, {age} år, har redan {pts} poäng på {gp} matcher.", "Som {age}-åring har {player} {g} mål och {pts} poäng.", "{player}: {pts} poäng på sina första {gp} matcher."],
               "r": ["Han spelar med ett lugn långt över sin ålder.", "Det är lagets framtid."]},
    "growth": {"s": ["En utvecklingshistoria.", "En spelare har tagit ett kliv i år.", "Scouterna har märkt något."],
               "p": ["{player} har gått från {from_ovr} till {to_ovr} i totalbetyg sedan säsongsstarten.", "Betyget för {player} har stigit med {delta}, till {to_ovr}.", "{player} började året på {from_ovr}. Nu är han en {to_ovr}."],
               "r": ["Tränarstaben gör ett bra jobb.", "Ett sånt kliv ändrar laguppställningen."]},
    "coach_record": {"s": ["Vi pratar om tränaren.", "Mannen bakom båset hos {team}."],
                     "p": ["{coach} har lett {team} till {record}.", "Under {coach} står {team} på {record}.", "Facit för {coach} i år: {record}."],
                     "r": ["Omklädningsrummet tror på honom.", "Hans system är tydligt."]},
    "team_form": {"s": ["Hur har {team} spelat på sistone?", "Formen för {team}."],
                  "p": ["{team} har gått {last10} de senaste tio matcherna.", "Senaste tio för {team}: {last10}.", "{team} står på {last10} på tio matcher."],
                  "r": ["Det säger mycket om stämningen.", "Formen spelar roll i den här ligan."]},
    "shots_tonight": {"s": ["Titta på skotten i kväll.", "Vem styr spelet hittills?"],
                      "p": ["{team} leder skottstatistiken mot {opp} med {a} mot {b}.", "Skott på mål: {team} {a}, {opp} {b}.", "Det är {a} skott mot {b} till {team}."],
                      "r": ["Trycket ger utdelning till slut.", "Målvakten i andra änden har fullt upp."]},
    "headline": {"s": ["Såg du nyheten i veckan?", "Det pratas om det i hela ligan."],
                 "p": ["{headline}.", "Rubriken: {headline}.", "Kort sagt: {headline}."],
                 "r": ["Det blir snack i omklädningsrummet.", "Vi får se hur det slutar."]},
    "hist_cups": {"s": ["Lite historia.", "En historisk notis om klubben."],
                  "p": ["{team} har vunnit Stanley Cup {cups} gånger.", "{cups} Stanley Cup-titlar i {team}s historia.", "Den här klubben har {cups} mästerskap."],
                  "r": ["Banderollerna berättar historien.", "Fansen här förväntar sig mycket."]},
    "hist_drought": {"s": ["En historisk notis som svider här.", "Fortfarande på jakt efter den första."],
                     "p": ["{team} har varit med i ligan sedan {first_season} och väntar fortfarande på sin första titel.", "Ingen Stanley Cup än för {team}, med sedan {first_season}.", "Sedan {first_season} har {team} drömt om sin första titel."],
                     "r": ["Den dagen det händer exploderar arenan.", "Det hoppet får folk att komma tillbaka."]},
    "atmosphere": {"s": ["Lyssna på publiken.", "Bra stämning i {city} i kväll."],
                   "p": ["Den här publiken kan sin hockey, den reagerar på de små sakerna.", "Varje tackling får publiken att jubla.", "De ställer sig upp för en befrielse. Det är en hockeystad."],
                   "r": ["Jag älskar att kommentera i sådana arenor.", "Det är därför man spelar."]},
    "road_trip": {"s": ["Vi pratar resor.", "Säsongens slit."],
                  "p": ["Tidiga flyg, sena ankomster, och ändå ska benen vara med.", "Hotell, arena, buss, flyg. Det är resandet som sliter.", "En andra match på två dagar efter ett sent flyg är det tuffaste testet."],
                  "r": ["Och vi sitter här uppe och dömer.", "Därför är bredden så viktig."]},
    "booth": {"s": ["Jag måste säga, kaffet i kväll...", "Min producent vinkar åt mig."],
              "p": ["Starkt nog för att klara en förlängning.", "Han vill att vi pratar om matchen. Vilken idé.", "Någon beställde soppan igen. Modigt."],
              "r": ["Tillbaka till hockeyn.", "Okej, ögonen på isen."]},
}
SV_CUTIN = ["Vänta lite.", "Oj, nu händer det.", "Vänta, vänta.", "Håll den tanken."]
SV_RESUME = ["Hur som helst, tillbaka till {topic}.", "Var var jag? Just det, {topic}."]

STORIES: Dict[str, Dict[str, Dict[str, List[str]]]] = {"en": EN, "fr": FR, "es": ES, "de": DE, "sv": SV}
CUTIN: Dict[str, List[str]] = {"en": EN_CUTIN, "fr": FR_CUTIN, "es": ES_CUTIN, "de": DE_CUTIN, "sv": SV_CUTIN}
RESUME: Dict[str, List[str]] = {"en": EN_RESUME, "fr": FR_RESUME, "es": ES_RESUME, "de": DE_RESUME, "sv": SV_RESUME}
GENERIC_CATS = frozenset({"atmosphere", "road_trip", "culture", "booth"})
SIM_CATS = frozenset(k for k in EN if not k.startswith("hist_") and k not in GENERIC_CATS)


def dataset_counts() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lang, cats in STORIES.items():
        composed = sum(len(c["s"]) * len(c["p"]) for c in cats.values())
        lines = sum(len(c["s"]) + len(c["p"]) + len(c["r"]) for c in cats.values())
        by = {"sim": 0, "history": 0, "generic": 0}
        for k, c in cats.items():
            grp = "history" if k.startswith("hist_") else ("generic" if k in GENERIC_CATS else "sim")
            by[grp] += len(c["s"]) * len(c["p"])
        out[lang] = {"categories": len(cats), "composed_exchanges": composed, "authored_lines": lines, "by_group": by}
    return out


def _slots(t: str) -> List[str]:
    return [f for _, f, _, _ in string.Formatter().parse(t) if f]


class _Safe(dict):
    def __missing__(self, k: str) -> str:
        raise KeyError(k)


def _fill(t: str, slots: Dict[str, Any]) -> Optional[str]:
    try:
        return t.format_map(_Safe(slots))
    except (KeyError, ValueError, IndexError):
        return None


# ════════════════════════════════════════════════════════════════════ facts (live session data only)

def _ovr(p: Any) -> Optional[int]:
    """Displayed 0-99 overall (same number the roster screens show)."""
    try:
        from services import franchise_sim as fs

        v = float(fs._player_display_ovr99(p))
        return int(round(v)) if v > 0 else None
    except Exception:
        try:
            v = getattr(p, "ovr", None)
            v = float(v() if callable(v) else v)
            return int(round(v * 99.0 if v <= 1.5 else v))
        except Exception:
            return None


def _name(p: Any) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", "") or "")


def _pid(p: Any) -> str:
    return str(getattr(p, "id", "") or "")


def _team_label(tm: Any) -> str:
    return str(getattr(tm, "city", "") or getattr(tm, "name", "") or "")


def _abbr(tm: Any) -> str:
    try:
        from services import franchise_sim as fs

        return str(fs._franchise_team_abbrev(tm))
    except Exception:
        return str(getattr(tm, "abbrev", "") or getattr(tm, "abbr", ""))[:3].upper()


def build_facts(session: Any, home: Any, away: Any, home_id: str, away_id: str, *, coach: str = "", user_tid: str = "") -> List[Dict[str, Any]]:
    """Facts for the two clubs playing (plus league context). Each: cat, key, slots, topic, players, weight."""
    facts: List[Dict[str, Any]] = []
    pss: Dict[str, Dict[str, Any]] = dict(getattr(session, "player_season_stats", None) or {})
    teams = {str(home_id): home, str(away_id): away}

    def add(cat: str, key: str, slots: Dict[str, Any], topic: str, players: Optional[List[str]] = None, weight: float = 1.0) -> None:
        # a count of exactly 1 would read "1 goals": drop that slot so only templates without it qualify
        slots = {k: v for k, v in slots.items() if not (k in _PLURAL_SLOTS and isinstance(v, int) and v == 1)}
        facts.append({"cat": cat, "key": f"{cat}:{key}", "slots": slots, "topic": topic, "players": players or [], "weight": weight})

    rows_by_team: Dict[str, List[Tuple[Any, Dict[str, Any]]]] = {}
    for tid, tm in teams.items():
        lst = []
        for p in list(getattr(tm, "roster", None) or []):
            r = pss.get(_pid(p))
            if r and str(r.get("team_id", tid)) == tid:
                lst.append((p, r))
        rows_by_team[tid] = lst
    for tid, tm in teams.items():
        T = _team_label(tm)
        sk = [(p, r) for p, r in rows_by_team[tid] if str(r.get("position", "")) != "G" and int(r.get("gp", 0) or 0) >= 3]
        if sk:
            p, r = max(sk, key=lambda x: (int(x[1].get("pts", 0)), int(x[1].get("g", 0))))
            if int(r.get("pts", 0)) >= 3:
                add("team_leader", _pid(p), {"team": T, "player": _name(p), "pts": int(r["pts"]), "g": int(r.get("g", 0)), "a": int(r.get("a", 0)), "gp": int(r["gp"])}, _name(p), [_pid(p)], 1.4)
            p, r = max(sk, key=lambda x: int(x[1].get("g", 0)))
            gp = int(r["gp"])
            if int(r.get("g", 0)) >= 4 and gp >= 8:
                pace = int(round(int(r["g"]) * 82.0 / gp))
                add("goal_pace", _pid(p), {"player": _name(p), "g": int(r["g"]), "gp": gp, "pace": pace}, _name(p), [_pid(p)], 1.2)
            p, r = max(sk, key=lambda x: int(x[1].get("ppg", 0)))
            if int(r.get("ppg", 0)) >= 3:
                add("pp_scorer", _pid(p), {"player": _name(p), "ppg": int(r["ppg"])}, _name(p), [_pid(p)])
            p, r = max(sk, key=lambda x: int(x[1].get("hit", 0)))
            if int(r.get("hit", 0)) >= 15:
                add("physical", _pid(p), {"player": _name(p), "team": T, "hits": int(r["hit"])}, _name(p), [_pid(p)], 0.8)
            p, r = max(sk, key=lambda x: int(x[1].get("blk", 0)))
            if int(r.get("blk", 0)) >= 10:
                add("blocker", _pid(p), {"player": _name(p), "team": T, "blk": int(r["blk"])}, _name(p), [_pid(p)], 0.8)
            ranked = sorted(sk, key=lambda x: -int(x[1].get("pts", 0)))
            ovr_rank = sorted(sk, key=lambda x: -(_ovr(x[0]) or 0))
            for i, (p, r) in enumerate(ranked[:6]):
                o = _ovr(p)
                orank = next((j for j, (q, _) in enumerate(ovr_rank) if q is p), 0)
                if o and orank - i >= 4 and int(r.get("pts", 0)) >= 5:
                    add("outperform", _pid(p), {"player": _name(p), "ovr": o, "pts": int(r["pts"]), "rank": _ORD_EN[i + 1] if i + 1 < len(_ORD_EN) else str(i + 1)}, _name(p), [_pid(p)])
            for p, r in sk:
                age = int(getattr(getattr(p, "identity", None), "age", 99) or 99)
                if age <= 21 and int(r.get("pts", 0)) >= 4:
                    add("rookie", _pid(p), {"player": _name(p), "age": age, "pts": int(r["pts"]), "g": int(r.get("g", 0)), "gp": int(r["gp"])}, _name(p), [_pid(p)], 1.1)
                c = getattr(p, "contract", None)
                yrs = getattr(c, "years_remaining", None) if c is not None else None
                if yrs == 1 and int(r.get("pts", 0)) >= 6:
                    add("contract_year", _pid(p), {"player": _name(p), "pts": int(r["pts"])}, _name(p), [_pid(p)], 0.9)
        for p in list(getattr(tm, "roster", None) or []):
            st = getattr(p, "season_start_ovr", None)
            now = _ovr(p)
            try:
                st = float(st) if st is not None else None
                st = int(round(st * 99.0 if st is not None and st <= 1.5 else st)) if st is not None else None
            except (TypeError, ValueError):
                st = None
            if st and now:
                if now - st >= 3:
                    add("growth", _pid(p), {"player": _name(p), "from_ovr": st, "to_ovr": now, "delta": now - st}, _name(p), [_pid(p)], 1.3)
                elif st - now >= 3:
                    add("decline", _pid(p), {"player": _name(p), "from_ovr": st, "to_ovr": now, "delta": st - now}, _name(p), [_pid(p)], 0.7)
        gl = [(p, r) for p, r in rows_by_team[tid] if str(r.get("position", "")) == "G" and int(r.get("goalie_shots_against", 0) or r.get("shots_against", 0) or 0) >= 60]
        if gl:
            p, r = max(gl, key=lambda x: int(x[1].get("goalie_shots_against", 0) or x[1].get("shots_against", 0)))
            sa = int(r.get("goalie_shots_against", 0) or r.get("shots_against", 0))
            ga = int(r.get("goalie_ga", 0) or r.get("ga", 0))
            svp = (sa - ga) / sa if sa else 0.0
            add("goalie_form", _pid(p), {"team": T, "goalie": _name(p), "w": int(r.get("w", 0)), "l": int(r.get("l", 0)), "svp": f".{int(round(svp * 1000)):03d}"}, _name(p), [_pid(p)], 1.1)
    # streaks / droughts from game results (team games in order)
    results = list(getattr(session, "game_results", None) or [])
    for tid in teams:
        games = [g for g in results if str(g.get("home_id")) == tid or str(g.get("away_id")) == tid]
        if len(games) < 5:
            continue
        pts_by_game: List[Dict[str, int]] = []
        goals_by_game: List[Dict[str, int]] = []
        for g in games:
            pg: Dict[str, int] = {}
            gg: Dict[str, int] = {}
            for se in g.get("scoring_events") or []:
                if str(se.get("for_team_id")) != tid:
                    continue
                sc = str(se.get("scorer_id") or "")
                pg[sc] = pg.get(sc, 0) + 1
                gg[sc] = gg.get(sc, 0) + 1
                for a in se.get("assist_ids") or []:
                    pg[str(a)] = pg.get(str(a), 0) + 1
            pts_by_game.append(pg)
            goals_by_game.append(gg)
        for p, r in rows_by_team[tid]:
            pid = _pid(p)
            if str(r.get("position", "")) == "G":
                continue
            n = 0
            for pg in reversed(pts_by_game):
                if pg.get(pid):
                    n += 1
                else:
                    break
            if n >= 3:
                add("point_streak", pid, {"player": _name(p), "n": n}, _name(p), [pid], 1.5)
            if int(r.get("g", 0)) >= 3:
                d = 0
                for gg in reversed(goals_by_game):
                    if gg.get(pid):
                        break
                    d += 1
                if d >= 6 and d < len(goals_by_game):
                    add("slump", pid, {"player": _name(p), "n": d, "g": int(r["g"])}, _name(p), [pid], 0.8)
    # meetings between the two clubs this season
    meet = [g for g in results if {str(g.get("home_id")), str(g.get("away_id"))} == {str(home_id), str(away_id)}]
    if meet:
        g = meet[-1]
        hg, ag = int(g.get("home_goals", 0) or 0), int(g.get("away_goals", 0) or 0)
        win = teams.get(str(g.get("winner_id") or (g.get("home_id") if hg > ag else g.get("away_id"))))
        if win is not None:
            add("last_meeting", str(g.get("game_id") or g.get("day")), {"winner": _team_label(win), "score": f"{max(hg, ag)}–{min(hg, ag)}", "day": int(g.get("day", 0) or 0)}, _team_label(win))
        if len(meet) >= 2:
            w = sum(1 for x in meet if str(x.get("winner_id")) == str(home_id))
            add("season_series", f"{len(meet)}", {"team": _team_label(home), "w": w, "l": len(meet) - w, "n": len(meet)}, _team_label(home))
    # standings, race, form, splits, coach
    try:
        rows = list(session.standings.as_table_rows())
    except Exception:
        rows = []
    by_tid = {str(r.get("team_id")): r for r in rows}
    for tid, tm in teams.items():
        r = by_tid.get(tid)
        if not r or int(r.get("gp", 0) or 0) < 5:
            continue
        T = _team_label(tm)
        div = [x for x in rows if x.get("division") == r.get("division")]
        pos = 1 + next((i for i, x in enumerate(div) if str(x.get("team_id")) == tid), 0)
        add("standings", f"{tid}:{r['gp']}", {"team": T, "pos": _ORD_EN[pos] if pos < len(_ORD_EN) else str(pos), "pos_n": pos, "division": str(r.get("division")), "pts": int(r.get("points", 0)), "gp": int(r["gp"])}, T)
        conf = sorted([x for x in rows if x.get("conference") == r.get("conference")], key=lambda x: -int(x.get("points", 0)))
        if len(conf) >= 9:
            cut = int(conf[7].get("points", 0))
            nxt = int(conf[8].get("points", 0))
            mine = int(r.get("points", 0))
            rank = next((i for i, x in enumerate(conf) if str(x.get("team_id")) == tid), 99)
            if rank < 8 and mine - nxt >= 1:
                add("race_in", f"{tid}:{r['gp']}", {"team": T, "delta": mine - nxt}, T)
            elif rank >= 8 and cut - mine >= 1:
                add("race_out", f"{tid}:{r['gp']}", {"team": T, "delta": cut - mine}, T)
        if r.get("last_10"):
            add("team_form", f"{tid}:{r['gp']}", {"team": T, "last10": str(r["last_10"])}, T, weight=0.9)
        if r.get("goals_for_per_game") is not None:
            add("scoring_rates", f"{tid}:{r['gp']}", {"team": T, "gfpg": f"{float(r['goals_for_per_game']):.1f}", "gapg": f"{float(r.get('goals_against_per_game', 0)):.1f}", "diff": str(r.get("goal_diff_label") or r.get("goal_diff"))}, T, weight=0.8)
        if r.get("home_record") and r.get("away_record"):
            add("home_road", f"{tid}:{r['gp']}", {"team": T, "home": str(r["home_record"]), "away": str(r["away_record"])}, T, weight=0.8)
        if coach and tid == str(user_tid):
            add("coach_record", f"{tid}:{r['gp']}", {"coach": coach, "team": T, "record": str(r.get("record"))}, coach, weight=0.8)
    # league scoring race
    allsk = [r for r in pss.values() if str(r.get("position", "")) != "G" and int(r.get("gp", 0) or 0) >= 5]
    if len(allsk) >= 2:
        top = sorted(allsk, key=lambda r: -int(r.get("pts", 0)))[:2]
        lead_t = (getattr(session, "team_by_id", {}) or {}).get(str(top[0].get("team_id")))
        if lead_t is not None and int(top[0].get("pts", 0)) > 0:
            add("league_race", str(top[0].get("player_id")), {"player": str(top[0].get("name")), "team": _team_label(lead_t), "pts": int(top[0]["pts"]),
                                                             "gap": int(top[0]["pts"]) - int(top[1].get("pts", 0)), "second": str(top[1].get("name"))}, str(top[0].get("name")), weight=0.9)
    # injury returns (healthy now, missed games)
    for e in list(getattr(session, "injury_log_all", None) or [])[-200:]:
        tid = str(e.get("team_id"))
        if tid in teams and int(e.get("games_initial", 0) or 0) >= 3:
            pid = str(e.get("player_id"))
            p = next((q for q in getattr(teams[tid], "roster", []) if _pid(q) == pid), None)
            if p is not None and str(getattr(getattr(p, "health", None), "injury_status", "")).lower().endswith("healthy"):
                add("injury_return", f"{pid}:{e.get('day')}", {"player": _name(p), "games": int(e["games_initial"])}, _name(p), [pid], 0.9)
    # headlines (existing storyline engine) for these teams
    seen = set()
    for e in reversed(list(getattr(session, "storyline_events", None) or [])[-300:]):
        if str(e.get("team_id") or e.get("team")) in teams and e.get("headline") and e["headline"] not in seen and len(seen) < 6:
            seen.add(e["headline"])
            add("headline", str(abs(hash(e["headline"])) % 10**9), {"headline": str(e["headline"]).rstrip("."), "team": _team_label(teams[str(e.get("team_id") or e.get("team"))])}, "that story", weight=0.7)
    # save history
    try:
        from services.legacy_hall import rivalries

        for rv in (rivalries(session, tid=str(home_id), limit=50) or {}).get("rivalries") or []:
            if {str(x) for x in rv.get("teams") or []} == {str(home_id), str(away_id)}:
                inp = rv.get("inputs") or {}
                if int(inp.get("gp", 0) or 0) >= 4:
                    add("rivalry_record", str(inp.get("gp")), {"team": _team_label(home), "opp": _team_label(away), "gp": int(inp["gp"]), "n_one": int(inp.get("one_goal", 0)), "series": int(inp.get("series", 0))}, "the rivalry")
    except Exception:
        pass
    for h in list(getattr(session, "season_history", None) or [])[-3:]:
        if isinstance(h, dict) and h.get("champion_name") and h.get("season"):
            add("past_champion", str(h["season"]), {"champion": str(h["champion_name"]), "season": str(h["season"])}, str(h["champion_name"]), weight=0.6)
    # stable franchise history
    for tid, tm in teams.items():
        hist = TEAM_HISTORY.get(_abbr(tm))
        if not hist:
            continue
        first, prev, cups = hist
        T = _team_label(tm)
        if cups == 1:
            add("hist_one_cup", tid, {"team": T}, T, weight=0.5)
        elif cups > 1:
            add("hist_cups", tid, {"team": T, "cups": cups}, T, weight=0.5)
        else:
            add("hist_drought", tid, {"team": T, "first_season": first}, T, weight=0.5)
        if prev:
            add("hist_reloc", tid, {"team": T, "prev_city": prev, "first_season": first}, T, weight=0.4)
        add("hist_age", tid, {"team": T, "first_season": first}, T, weight=0.3)
    city = _team_label(home)
    for cat in ("atmosphere", "road_trip", "culture", "booth"):
        add(cat, cat, {"city": city}, "", weight=0.35)
    return facts


def live_facts(home_label: str, away_label: str, sog_home: int, sog_away: int, on_ice: List[Tuple[str, Dict[str, Any]]], t: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if sog_home + sog_away >= 12 and abs(sog_home - sog_away) >= 5:
        lead, opp, a, b = (home_label, away_label, sog_home, sog_away) if sog_home > sog_away else (away_label, home_label, sog_away, sog_home)
        out.append({"cat": "shots_tonight", "key": f"shots_tonight:{int(t // 600)}", "slots": {"team": lead, "opp": opp, "a": a, "b": b}, "topic": "the shot clock", "players": [], "weight": 1.0})
    for pid, row in on_ice:
        if int(row.get("pts", 0)) >= 6 and int(row.get("g", 0)) >= 2 and int(row.get("a", 0)) >= 2:
            out.append({"cat": "on_ice_line", "key": f"on_ice_line:{pid}", "slots": {"player": row["name"], "last": str(row["name"]).split()[-1], "g": int(row.get("g", 0)), "a": int(row.get("a", 0)), "pts": int(row["pts"]), "gp": int(row.get("gp", 0))},
                        "topic": row["name"], "players": [pid], "weight": 0.5})
    return out


def select(lang: str, facts: List[Dict[str, Any]], told: Dict[str, int], game_no: int, rng: random.Random, *,
           on_ice: Optional[set] = None, avoid_cats: Optional[set] = None, generic_ok: bool = True,
           used_lines: Optional[set] = None, cat_counts: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
    """Pick one fact + template. Returns {key, cat, lines: [3 strings], topic} or None."""
    tables = STORIES.get(lang) or STORIES["en"]
    cands: List[Tuple[float, Dict[str, Any]]] = []
    for f in facts:
        cat = f["cat"]
        if cat not in tables or (avoid_cats and cat in avoid_cats):
            continue
        if cat in GENERIC_CATS and not generic_ok:
            continue
        if cat_counts and cat_counts.get(cat, 0) >= 3:
            continue
        last = told.get(f["key"])
        if last is not None and game_no - last < 10:
            continue
        w = float(f.get("weight", 1.0))
        if on_ice and any(p in on_ice for p in f.get("players") or []):
            w *= 2.0
        cands.append((w, f))
    rng.shuffle(cands)
    cands.sort(key=lambda x: -x[0] * rng.uniform(0.6, 1.4))
    for _w, f in cands[:12]:
        tb = tables[f["cat"]]
        combos = [(i, j) for i in range(len(tb["s"])) for j in range(len(tb["p"]))]
        rng.shuffle(combos)
        for i, j in combos:
            used = used_lines if used_lines is not None else set()
            if ("s", f["cat"], i) in used or ("p", f["cat"], j) in used:
                continue
            s = _fill(tb["s"][i], f["slots"])
            p = _fill(tb["p"][j], f["slots"])
            if s is None or p is None:
                continue
            reacts = [x for k, x in ((k, _fill(r, f["slots"])) for k, r in enumerate(tb["r"])) if x and ("r", f["cat"], k) not in used]
            r = rng.choice(reacts) if reacts else None
            if used_lines is not None:
                used_lines.update({("s", f["cat"], i), ("p", f["cat"], j)})
                if r is not None:
                    used_lines.add(("r", f["cat"], [_fill(x, f["slots"]) for x in tb["r"]].index(r)))
            lines = [s, p] + ([r] if r else [])
            return {"key": f["key"], "cat": f["cat"], "tid": f"{f['cat']}:{i}:{j}", "lines": lines, "topic": f.get("topic") or "", "slots": dict(f["slots"])}
    return None


# ════════════════════════════════════════════════════════════════════ ROUND 4: colour + history expansion
# Colour categories: some always eligible ({city}, {team}/{opp} from the matchup), some only when the live game supports them.
EN.update({
    "bit_coffee": {
        "s": ["You've been quiet over there.", "I see you brought the giant thermos again.", "Did you get the good coffee or the press box coffee?",
              "I notice you've switched to decaf.", "Every year you swear you'll cut back on the caffeine."],
        "p": ["Third cup. I'm pacing myself for overtime.", "Somebody has to keep this booth awake through the second period.",
              "Press box coffee. I've built up a tolerance over the years.", "Doctor's orders. I don't want to talk about it.", "And every year I make it to about the first intermission."],
        "r": ["Your producer is going to cut your budget.", "One of these nights we'll get a decent espresso machine up here.", "Fair enough. Eyes on the ice."],
    },
    "bit_predictions": {
        "s": ["How's your prediction from before the game looking?", "You said something in the pregame I want to bring up.",
              "Remember the score you picked this afternoon?", "You've got a perfect record on predictions this week, right?", "Let's check on the booth pick."],
        "p": ["Let's just say I'm sticking with it until the final horn.", "I said it would be a goalie duel. We'll see.",
              "I'm not going to repeat it on air. That's how these things go wrong.", "Perfect is a strong word. Let's call it optimistic.", "Still alive. Barely, but alive."],
        "r": ["I'll hold you to that one.", "Bold. I like it.", "We'll revisit that at the final buzzer."],
    },
    "bit_rivalry_booth": {
        "s": ["You and I have a little side bet on this season, don't we?", "Our booth rivalry needs an update.",
              "Who won the pickup game last week? Remind the people.", "The crew still talks about your skating at that charity game.", "I hear you've been working on your wrist shot."],
        "p": ["Loser buys dinner on the next road trip. I'm feeling good about it.", "I'm up two games to one. Nobody asked, but I'm up.",
              "We agreed not to discuss it. Moving on.", "I scored. That's all anybody needs to remember.", "Strictly theoretical. I'm a student of the game."],
        "r": ["We'll settle it in the summer league.", "I want a recount.", "The tape will show what really happened."],
    },
    "crowd_arena": {
        "s": ["The sound in this building tonight.", "You can feel the building shake a little on some of these hits.",
              "{city} shows up for hockey.", "Look at the seats behind the far net.", "There's a group in the upper bowl that hasn't stopped chanting."],
        "p": ["It gets loud fast, and it stays loud. That's a real hockey crowd.", "The old barns had that, and this place has found it too.",
              "Weeknight, cold out, and the building is still full.", "Every jersey you can think of, old ones, new ones, a few that should be retired.", "They've got drums, they've got lungs, and they've got opinions."],
        "r": ["Players feed off that, believe me.", "That's a home-ice advantage you can hear.", "Love it."],
    },
    "crowd_live": {
        "s": ["Listen to them now.", "This crowd hasn't sat down since that last goal.", "Feel that energy?", "The fans know what's at stake tonight.", "You can tell the building is into it."],
        "p": ["{team} has given them something to cheer about tonight.", "When {team} is up {lead_text}, this place gets rowdy.",
              "Every time {team} touches the puck, they rise a little.", "They've been loud since the opening shift, and the score is helping.", "{team} on top and the place is rocking."],
        "r": ["That's the fun part of the job.", "Keep it up, folks.", "You'll hear them in the dressing room."],
    },
    "road_life": {
        "s": ["How was the flight in?", "Road trips this time of year are something else.", "You know what nobody tells you about travel?",
              "Long trip for somebody this week.", "The schedule has been brutal lately."],
        "p": ["Delayed two hours, then a bus ride, then straight to the rink. Glamorous life.", "Four cities in six nights. You stop knowing which hotel you're in.",
              "The hardest part is the time zones. Your body never knows when dinner is.", "Players will tell you the third road game in a row is the toughest one.", "You learn to sleep anywhere. Airports, buses, a press box chair."],
        "r": ["And they still have to bring their legs every night.", "That's why depth matters this time of year.", "Somebody get these guys a nap."],
    },
    "rink_food": {
        "s": ["Did you try the new thing at the concession stand?", "The press box food tonight deserves a mention.", "Big question: best rink snack?",
              "You were gone a long time at the intermission.", "I smelled something good coming up the stairs."],
        "p": ["The loaded fries. I regret nothing.", "Let's say the soup is ambitious.", "A hot dog and a coffee. Classic for a reason.",
              "The line was long, but the poutine was worth it.", "Somebody brought homemade cookies for the crew. Bless them."],
        "r": ["Back to hockey before we get hungry again.", "Save me some next time.", "The diet starts tomorrow."],
    },
    "superstitions": {
        "s": ["Hockey players and their superstitions.", "You have any game-day rituals?", "I saw something in warmups I have to ask about.",
              "Every dressing room has that one guy.", "Ever notice how goalies talk to their posts?"],
        "p": ["Same stick tape, same pregame meal, same seat on the bus. Don't touch any of it.", "Left skate first, always. I did it in house league and I still do it.",
              "One guy taps every teammate's pads in the same order before the anthem.", "There's always someone who refuses to shave during a winning streak.", "They'll tell you it's just focus. It's not just focus."],
        "r": ["Whatever works.", "I'm not saying it works, but I'm not saying it doesn't.", "Don't mess with a streak."],
    },
    "gear_talk": {
        "s": ["Look at the sticks these guys use now.", "Equipment has come a long way.", "Have you held one of the modern goalie pads?",
              "The skates today are something else.", "Remember wooden sticks?"],
        "p": ["Light as a feather, and they flex like a fishing rod.", "When I started, the shin pads were basically magazines and tape.",
              "They're lighter than they look, and somehow they cover more net.", "Custom fit, heat molded. We used to wear our older brother's.", "Heavy, and you'd break three a week. The stick boys loved us."],
        "r": ["Better gear, faster game.", "The game keeps changing.", "Kids today have no idea."],
    },
    "old_time": {
        "s": ["Hockey today versus back in the day.", "You ever wish you played in a different era?", "The game is faster than it's ever been.",
              "Old-timers love to tell you how tough it used to be.", "Shift lengths have changed so much."],
        "p": ["Shifts used to be two minutes. Now it's forty seconds and back to the bench.", "Back then you could hook a guy for a whole shift and nobody blinked.",
              "Everybody can skate now. The fourth line can fly.", "And they're half right. It was tough. It was also slow.", "Systems everywhere now. We just chased the puck."],
        "r": ["Better hockey, I'll say that much.", "Every era thinks it was the best.", "I'll take this version."],
    },
    "kids_hockey": {
        "s": ["You know what this reminds me of?", "When did you first lace them up?", "Early morning practices as a kid.",
              "Think about all the kids watching tonight.", "Outdoor rinks. You miss them?"],
        "p": ["Six in the morning, cold rink, my dad with a coffee in the stands. Every Saturday.", "Pond hockey until it got dark, then a little longer.",
              "I was the kid who spent more time on his backside than on his skates.", "Somebody out there is going to be on this ice in fifteen years.", "Every winter. Shovel the snow, play until your feet go numb."],
        "r": ["That's where the love of the game starts.", "Best times of your life.", "The parents are the real heroes."],
    },
    "weather": {
        "s": ["Cold one in {city} tonight.", "How's the weather been treating you here?", "Did you see the forecast for tomorrow?",
              "Somebody said it might snow later.", "The walk from the parking lot tonight."],
        "p": ["Perfect hockey weather. You want it cold when you come to the rink.", "Windy, grey, and you couldn't ask for a better night for a game.",
              "Whatever it is, it's warmer in here with this crowd.", "If it snows, the drive home will be an adventure.", "Brisk. Let's call it brisk."],
        "r": ["Bring a scarf.", "That's {city} for you.", "Nobody's complaining in here."],
    },
    "form_banter": {
        "s": ["{team} has been on a bit of a run.", "Momentum is a real thing in this league.", "Confidence looks different from team to team.",
              "There's a different feel around {team} lately.", "You can see it in how {team} is skating."],
        "p": ["{team} is {last10} in the last ten, and it shows in the way they move the puck.", "Winning {wins} of the last ten does wonders for a room.",
              "Ten games, {wins} wins for {team}. That changes how you show up to the rink.", "{team} at {last10} lately. The bench is relaxed.", "With {wins} wins in ten, {team} is playing like it expects to win."],
        "r": ["Winning is contagious.", "That's when the hockey gets fun.", "You can't fake that."],
    },
})

# History from the save + repo team data (eligible on a fresh save too)
EN.update({
    "hist_years": {
        "s": ["How long has {team} been around?", "Little perspective on {team} for you.", "Here's a franchise note.",
              "Between whistles, a history nugget.", "Think about all the seasons in that building."],
        "p": ["This is season number {years} for {team}.", "{team} is in year {years} as an NHL club.",
              "{years} seasons now since {team} joined the league.", "{team} is {years} years into its NHL story.", "{team} since {first_season}. That's {years} seasons of hockey."],
        "r": ["A lot of memories in that time.", "Generations of fans.", "Time flies."],
    },
    "hist_compare": {
        "s": ["Two very different franchise histories here tonight.", "Quick comparison of these two clubs.", "Here's something about this matchup.",
              "History buffs, this one's for you.", "These two franchises came into the league at different times."],
        "p": ["{old_team} dates back to {old_year}. {new_team} arrived in {new_year}.", "{old_team} has {gap} more years of history than {new_team}.",
              "{new_team} joined in {new_year}, {gap} years after {old_team}.", "{old_team} was already {gap} years old when {new_team} played its first game.", "{old_year} for {old_team}, {new_year} for {new_team}."],
        "r": ["Old school against new school.", "Different roots, same goal.", "History doesn't play the game, but it's fun to think about."],
    },
    "hist_matchup_cups": {
        "s": ["Banner count, if you're keeping track.", "Total championships between these two?", "A little trophy talk.",
              "Two franchises, one question: who has more hardware?", "Here's one for the trivia crowd."],
        "p": ["Between them, {team} and {opp} have {total} Stanley Cups.", "{team} has {cups_a}, {opp} has {cups_b}.",
              "Add them up and it's {total} championships in this building tonight.", "{cups_a} for {team}, {cups_b} for {opp}.", "That's {total} Cups combined between the two clubs."],
        "r": ["Not that it matters on the ice tonight.", "Some fans will remind you of that.", "Banners don't score goals, though."],
    },
    "save_meetings": {
        "s": ["These two have met plenty in this league.", "Season series history in our books.", "The record between them this year.",
              "Head to head, the books say.", "How have these meetings gone?"],
        "p": ["{n} meetings so far, with {team} winning {w}.", "{team} has taken {w} of {n} against {opp}.",
              "It's {w} wins for {team} in {n} games against {opp}.", "{opp} has {l} of the {n} meetings.", "Through {n} games, {team} {w}, {opp} {l}."],
        "r": ["Every game in this series has had something.", "That's why tonight matters.", "They don't like each other much."],
    },
    "save_last_result": {
        "s": ["Let's go back to the last meeting.", "Last time out between these two.", "The previous game between them.",
              "Remember how the last one ended?", "Recent history between these clubs."],
        "p": ["{winner} won {score}, and the other room hasn't forgotten.", "It was {winner}, {score}.",
              "That finished {score} in favour of {winner}.", "{winner} came out on top, {score}.", "A {score} final, {winner} the winner."],
        "r": ["Different night, though.", "Revenge is a strong motivator.", "Both coaches will have talked about it."],
    },
    "franchise_memory": {
        "s": ["Some longer-term history for this club.", "The franchise's recent record in the playoffs.", "Here's a team history note from our books.",
              "Let's look at the bigger picture for {team}.", "Years of history in one stat."],
        "p": ["{team} has {apps} playoff appearances in our records.", "{team} has won {series} playoff series over the years in our books.",
              "It's been {since} seasons since {team} last made the playoffs.", "{apps} trips to the playoffs for {team}.", "{series} series wins on the franchise ledger."],
        "r": ["History matters to the fans.", "That's the bar this group is chasing.", "Every season adds a chapter."],
    },
})

# Natural bridges, replies, hand-backs (English)
BRIDGES: Dict[str, Dict[str, List[str]]] = {
    "en": {"colour": ["You know what this reminds me of?", "While we've got a second,", "Can I tell you something?", "Not to change the subject, but", "Totally unrelated, but"],
           "crowd": ["Speaking of the crowd,", "Listen to this place."],
           "history": ["Quick history lesson while they reset.", "While they line up, a little history.", "Here's something for the history buffs.", "Bit of background for you."],
           "sim": ["Since things have settled down,", "Quick stat while we've got a lull.", "While it's quiet out there,", "Let me get this in."]},
    "fr": {"colour": ["Vous savez à quoi ça me fait penser?", "En parlant de la foule,", "Pendant qu'on a une seconde,"],
           "history": ["Petite leçon d'histoire pendant l'arrêt.", "Un peu d'histoire pendant qu'ils se replacent."],
           "sim": ["Pendant que c'est calme,", "Une petite statistique en passant."]},
    "es": {"colour": ["¿Sabes a qué me recuerda esto?", "Hablando de la afición,", "Ya que tenemos un momento,"],
           "history": ["Una pequeña lección de historia mientras se colocan.", "Un poco de historia mientras tanto."],
           "sim": ["Ahora que está tranquilo,", "Un dato rápido aprovechando la pausa."]},
    "de": {"colour": ["Wissen Sie, woran mich das erinnert?", "Apropos Publikum,", "Wo wir gerade einen Moment haben,"],
           "history": ["Kleine Geschichtsstunde, während sie sich sortieren.", "Ein bisschen Geschichte zwischendurch."],
           "sim": ["Jetzt, wo es ruhig ist,", "Kurz eine Zahl, solange es ruhig ist."]},
    "sv": {"colour": ["Vet du vad det här påminner mig om?", "På tal om publiken,", "Medan vi har en sekund,"],
           "history": ["Liten historielektion medan de ställer upp.", "Lite historia så länge."],
           "sim": ["Nu när det lugnat sig,", "En snabb siffra medan det är lugnt."]},
}
REPLIES_BY_FAMILY: Dict[str, Dict[str, List[str]]] = {
    "en": {"sim": ["Is that right?", "No kidding.", "That's impressive.", "I hadn't noticed that.", "Good catch.", "That's a real number."],
           "history": ["I didn't know that.", "Huh, I had no idea.", "That's a long time.", "You'd remember that better than me.", "Some history there."],
           "colour": ["Ha. Classic.", "You would say that.", "I believe it.", "Same here, honestly.", "Of course you did.", "Can't argue with that."]},
}
REPLIES: Dict[str, List[str]] = {
    "en": ["Is that right?", "I didn't know that.", "No kidding."],
    "fr": ["Ah oui?", "Je ne savais pas.", "C'est logique.", "Tu as raison."],
    "es": ["¿En serio?", "No lo sabía.", "Tiene sentido.", "Tienes razón."],
    "de": ["Ach wirklich?", "Das wusste ich nicht.", "Das ergibt Sinn.", "Da hast du recht."],
    "sv": ["Jaså?", "Det visste jag inte.", "Det låter rimligt.", "Du har rätt."],
}
HANDBACKS: Dict[str, List[str]] = {
    "en": ["Anyway, {team} bringing it up ice.", "Back to it, {team} with the puck.", "Okay, {team} coming the other way.", "Meanwhile, {team} moving it out."],
    "fr": ["Bref, {team} remonte la rondelle.", "Revenons au jeu, {team} contrôle la rondelle."],
    "es": ["En fin, {team} sube el disco.", "Volvemos al juego, {team} con el disco."],
    "de": ["Jedenfalls, {team} baut auf.", "Zurück zum Spiel, {team} hat die Scheibe."],
    "sv": ["Hur som helst, {team} för upp pucken.", "Tillbaka till spelet, {team} har pucken."],
}

# ── fr / es / de / sv additions (≥50 composed each)
FR.update({
    "crowd_arena": {"s": ["Écoutez le bruit dans l'aréna.", "{city} répond présent pour le hockey.", "Regardez les gradins derrière le filet."],
                    "p": ["Ça devient bruyant vite, et ça reste bruyant.", "Un soir de semaine, il fait froid, et l'aréna est pleine.", "Des chandails de toutes les époques dans la foule."],
                    "r": ["Les joueurs s'en nourrissent.", "Un vrai avantage de la glace."]},
    "rink_food": {"s": ["Tu as goûté la nouveauté au casse-croûte?", "La nourriture de la galerie de presse ce soir...", "Meilleure collation d'aréna?"],
                  "p": ["La poutine. Aucun regret.", "Disons que la soupe est ambitieuse.", "Un hot-dog et un café. Un classique."],
                  "r": ["Revenons au hockey avant d'avoir faim.", "Le régime commence demain."]},
    "superstitions": {"s": ["Les joueurs et leurs superstitions.", "Tu as des rituels de jour de match?", "Chaque vestiaire a ce gars-là."],
                      "p": ["Même ruban, même repas, même siège dans l'autobus.", "Le patin gauche en premier, toujours.", "Il y en a toujours un qui refuse de se raser pendant une séquence."],
                      "r": ["Si ça marche...", "On ne touche pas à une séquence."]},
    "old_time": {"s": ["Le hockey d'aujourd'hui contre celui d'antan.", "Le jeu n'a jamais été aussi rapide.", "Les présences ont tellement changé."],
                 "p": ["Les présences duraient deux minutes. Maintenant, quarante secondes.", "Tout le monde patine maintenant, même le quatrième trio.", "Des systèmes partout. Avant, on courait après la rondelle."],
                 "r": ["Du meilleur hockey, je l'avoue.", "Chaque époque pense être la meilleure."]},
    "kids_hockey": {"s": ["Vous savez à quoi ça me fait penser?", "Les pratiques tôt le matin quand on était jeunes.", "Pensez aux enfants qui regardent ce soir."],
                    "p": ["Six heures du matin, aréna froide, mon père avec un café dans les gradins.", "Du hockey sur l'étang jusqu'à la noirceur.", "Un de ces jeunes jouera sur cette glace dans quinze ans."],
                    "r": ["C'est là que naît l'amour du jeu.", "Les parents sont les vrais héros."]},
    "hist_years": {"s": ["Depuis combien de temps {team} existe?", "Un peu de perspective sur {team}."],
                   "p": ["C'est la saison numéro {years} de la concession.", "{team} en est à sa {years}e saison dans la ligue.", "Depuis {first_season}, ça fait {years} saisons."],
                   "r": ["Beaucoup de souvenirs.", "Des générations de partisans."]},
    "hist_compare": {"s": ["Deux histoires bien différentes ce soir.", "Petite comparaison entre ces deux clubs."],
                     "p": ["{old_team} remonte à {old_year}, {new_team} est arrivé en {new_year}.", "{new_team} est arrivé {gap} ans après {old_team}.", "{old_year} pour {old_team}, {new_year} pour {new_team}."],
                     "r": ["La vieille école contre la nouvelle.", "Des racines différentes, le même objectif."]},
})
ES.update({
    "crowd_arena": {"s": ["Escuchen el ruido del pabellón.", "{city} responde cuando hay hockey.", "Miren las gradas detrás de la portería."],
                    "p": ["Se pone ruidoso rápido y se mantiene así.", "Entre semana, hace frío, y el pabellón está lleno.", "Camisetas de todas las épocas en la grada."],
                    "r": ["Los jugadores lo notan.", "Una ventaja de jugar en casa."]},
    "rink_food": {"s": ["¿Probaste lo nuevo del puesto de comida?", "La comida de la tribuna de prensa esta noche...", "¿El mejor bocado de pabellón?"],
                  "p": ["Las patatas cargadas. Sin arrepentimientos.", "Digamos que la sopa es ambiciosa.", "Un perrito caliente y un café. Un clásico."],
                  "r": ["Volvamos al hockey antes de tener hambre.", "La dieta empieza mañana."]},
    "superstitions": {"s": ["Los jugadores y sus supersticiones.", "¿Tienes algún ritual de día de partido?", "Todo vestuario tiene a ese jugador."],
                      "p": ["La misma cinta, la misma comida, el mismo asiento en el autobús.", "Siempre el patín izquierdo primero.", "Siempre hay uno que no se afeita durante una racha."],
                      "r": ["Si funciona...", "Una racha no se toca."]},
    "old_time": {"s": ["El hockey de hoy contra el de antes.", "El juego nunca ha sido tan rápido.", "Los turnos han cambiado muchísimo."],
                 "p": ["Los turnos duraban dos minutos. Ahora, cuarenta segundos.", "Ahora todos patinan, hasta la cuarta línea.", "Sistemas por todas partes. Antes perseguíamos el disco."],
                 "r": ["Mejor hockey, lo reconozco.", "Cada época se cree la mejor."]},
    "kids_hockey": {"s": ["¿Sabes a qué me recuerda esto?", "Los entrenamientos de madrugada de niño.", "Piensen en los niños que miran esta noche."],
                    "p": ["Seis de la mañana, pista fría, mi padre con un café en la grada.", "Hockey en el estanque hasta que oscurecía.", "Alguno de esos niños jugará en esta pista dentro de quince años."],
                    "r": ["Ahí nace el amor por el juego.", "Los padres son los verdaderos héroes."]},
    "hist_years": {"s": ["¿Cuánto tiempo lleva {team}?", "Un poco de perspectiva sobre {team}."],
                   "p": ["Es la temporada número {years} de la franquicia.", "{team} va por su temporada {years} en la liga.", "Desde {first_season}, son {years} temporadas."],
                   "r": ["Muchos recuerdos.", "Generaciones de aficionados."]},
    "hist_compare": {"s": ["Dos historias muy distintas esta noche.", "Una comparación rápida entre estos clubes."],
                     "p": ["{old_team} se remonta a {old_year}, {new_team} llegó en {new_year}.", "{new_team} llegó {gap} años después que {old_team}.", "{old_year} para {old_team}, {new_year} para {new_team}."],
                     "r": ["Vieja escuela contra nueva escuela.", "Raíces distintas, el mismo objetivo."]},
})
DE.update({
    "crowd_arena": {"s": ["Hören Sie den Lärm in der Halle.", "{city} ist da, wenn es um Eishockey geht.", "Schauen Sie auf die Ränge hinter dem Tor."],
                    "p": ["Es wird schnell laut, und es bleibt laut.", "Unter der Woche, draußen kalt, und die Halle ist voll.", "Trikots aus allen Epochen auf den Rängen."],
                    "r": ["Die Spieler ziehen Energie daraus.", "Ein echter Heimvorteil."]},
    "rink_food": {"s": ["Hast du das Neue am Imbiss probiert?", "Das Essen auf der Pressetribüne heute...", "Bester Hallensnack?"],
                  "p": ["Die beladenen Pommes. Keine Reue.", "Sagen wir, die Suppe ist ehrgeizig.", "Eine Bratwurst und ein Kaffee. Ein Klassiker."],
                  "r": ["Zurück zum Eishockey, bevor wir wieder Hunger bekommen.", "Die Diät beginnt morgen."]},
    "superstitions": {"s": ["Eishockeyspieler und ihr Aberglaube.", "Hast du Rituale am Spieltag?", "Jede Kabine hat so einen Spieler."],
                      "p": ["Gleiches Tape, gleiches Essen, gleicher Platz im Bus.", "Immer zuerst der linke Schlittschuh.", "Einer rasiert sich während einer Siegesserie nie."],
                      "r": ["Wenn es hilft...", "An einer Serie rührt man nicht."]},
    "old_time": {"s": ["Eishockey heute gegen früher.", "Das Spiel war nie schneller.", "Die Wechsel haben sich so verändert."],
                 "p": ["Früher dauerte ein Wechsel zwei Minuten. Heute vierzig Sekunden.", "Heute können alle laufen, sogar die vierte Reihe.", "Überall Systeme. Früher sind wir der Scheibe hinterhergejagt."],
                 "r": ["Besseres Eishockey, das gebe ich zu.", "Jede Epoche hält sich für die beste."]},
    "kids_hockey": {"s": ["Wissen Sie, woran mich das erinnert?", "Frühes Training als Kind.", "Denken Sie an die Kinder, die heute zuschauen."],
                    "p": ["Sechs Uhr morgens, kalte Halle, mein Vater mit einem Kaffee auf der Tribüne.", "Eishockey auf dem Teich, bis es dunkel wurde.", "Eines dieser Kinder steht in fünfzehn Jahren auf diesem Eis."],
                    "r": ["Da beginnt die Liebe zum Spiel.", "Die Eltern sind die wahren Helden."]},
    "hist_years": {"s": ["Wie lange gibt es {team} schon?", "Ein bisschen Perspektive zu {team}."],
                   "p": ["Das ist Saison Nummer {years} für die Franchise.", "{team} spielt seine {years}. Saison in der Liga.", "Seit {first_season}, das sind {years} Spielzeiten."],
                   "r": ["Viele Erinnerungen.", "Generationen von Fans."]},
    "hist_compare": {"s": ["Zwei sehr unterschiedliche Geschichten heute.", "Ein kurzer Vergleich der beiden Klubs."],
                     "p": ["{old_team} gibt es seit {old_year}, {new_team} kam {new_year} dazu.", "{new_team} kam {gap} Jahre nach {old_team} in die Liga.", "{old_year} für {old_team}, {new_year} für {new_team}."],
                     "r": ["Alte Schule gegen neue Schule.", "Unterschiedliche Wurzeln, gleiches Ziel."]},
})
SV.update({
    "crowd_arena": {"s": ["Lyssna på ljudet i arenan.", "{city} ställer upp när det är hockey.", "Titta på läktaren bakom målet."],
                    "p": ["Det blir högljutt snabbt, och det håller i sig.", "En vardagskväll, kallt ute, och arenan är full.", "Tröjor från alla epoker på läktaren."],
                    "r": ["Spelarna lever på det.", "En riktig hemmafördel."]},
    "rink_food": {"s": ["Har du testat det nya i kiosken?", "Maten i pressläktaren i kväll...", "Bästa arenasnacket?"],
                  "p": ["Pommesen med allt. Ångrar ingenting.", "Låt oss säga att soppan är ambitiös.", "En korv och en kaffe. En klassiker."],
                  "r": ["Tillbaka till hockeyn innan vi blir hungriga igen.", "Dieten börjar i morgon."]},
    "superstitions": {"s": ["Hockeyspelare och deras vidskepelse.", "Har du några ritualer på matchdagen?", "Varje omklädningsrum har den där killen."],
                      "p": ["Samma tejp, samma måltid, samma plats på bussen.", "Vänster skridsko först, alltid.", "Det finns alltid någon som inte rakar sig under en segersvit."],
                      "r": ["Om det funkar så...", "Man rör inte en svit."]},
    "old_time": {"s": ["Dagens hockey mot förr.", "Spelet har aldrig varit snabbare.", "Bytena har förändrats så mycket."],
                 "p": ["Förr var ett byte två minuter. Nu fyrtio sekunder.", "Alla kan åka skridskor nu, även fjärdekedjan.", "System överallt. Förr jagade vi bara pucken."],
                 "r": ["Bättre hockey, det erkänner jag.", "Varje epok tror att den var bäst."]},
    "kids_hockey": {"s": ["Vet du vad det här påminner mig om?", "Tidiga träningar som barn.", "Tänk på alla barn som tittar i kväll."],
                    "p": ["Klockan sex på morgonen, kall rink, pappa med en kaffe på läktaren.", "Hockey på dammen tills det blev mörkt.", "Ett av de där barnen spelar på den här isen om femton år."],
                    "r": ["Det är där kärleken till spelet börjar.", "Föräldrarna är de riktiga hjältarna."]},
    "hist_years": {"s": ["Hur länge har {team} funnits?", "Lite perspektiv på {team}."],
                   "p": ["Det här är säsong nummer {years} för klubben.", "{team} spelar sin säsong nummer {years} i ligan.", "Sedan {first_season} har det blivit {years} säsonger."],
                   "r": ["Många minnen.", "Generationer av fans."]},
    "hist_compare": {"s": ["Två väldigt olika historier i kväll.", "En snabb jämförelse mellan klubbarna."],
                     "p": ["{old_team} går tillbaka till {old_year}, {new_team} kom {new_year}.", "{new_team} kom {gap} år efter {old_team}.", "{old_year} för {old_team}, {new_year} för {new_team}."],
                     "r": ["Gamla skolan mot nya skolan.", "Olika rötter, samma mål."]},
})

# fix: never say "number one" for a shift; use the surname instead
EN["on_ice_line"]["s"] = ["He's on the ice right now, so let's talk about him.", "Look who's out there now.",
                          "There's a guy on this shift worth a mention.", "Keep an eye on {last} this shift."]

COLOUR_CATS = frozenset({"atmosphere", "road_trip", "culture", "booth", "bit_coffee", "bit_predictions", "bit_rivalry_booth", "crowd_arena", "crowd_live",
                         "road_life", "rink_food", "superstitions", "gear_talk", "old_time", "kids_hockey", "weather", "form_banter"})
HISTORY_CATS = frozenset({"hist_cups", "hist_drought", "hist_reloc", "hist_age", "hist_years", "hist_compare", "hist_matchup_cups",
                          "save_meetings", "save_last_result", "franchise_memory", "past_champion", "rivalry_record", "last_meeting", "season_series"})
GENERIC_CATS = COLOUR_CATS
FAMILY_TARGET = {"sim": 0.40, "colour": 0.30, "history": 0.30}
# combo-level telling (these facts are few and static, the wording is what must not repeat)
COMBO_KEYED = COLOUR_CATS | HISTORY_CATS


def family(cat: str) -> str:
    if cat in COLOUR_CATS:
        return "colour"
    if cat in HISTORY_CATS:
        return "history"
    return "sim"


def dataset_counts() -> Dict[str, Any]:  # noqa: F811 (round-4 version: by family)
    out: Dict[str, Any] = {}
    for lang, cats in STORIES.items():
        by = {"sim": 0, "colour": 0, "history": 0}
        for k, c in cats.items():
            by[family(k)] += len(c["s"]) * len(c["p"])
        out[lang] = {"categories": len(cats), "composed_exchanges": sum(by.values()),
                     "authored_lines": sum(len(c["s"]) + len(c["p"]) + len(c["r"]) for c in cats.values()), "by_family": by}
    return out


def extra_facts(session: Any, home: Any, away: Any, home_id: str, away_id: str) -> List[Dict[str, Any]]:
    """Round-4 facts: colour slots (city/teams/form) and history from repo data + the save."""
    out: List[Dict[str, Any]] = []
    H, A = _team_label(home), _team_label(away)
    city = H
    teams = {str(home_id): home, str(away_id): away}

    def add(cat: str, key: str, slots: Dict[str, Any], topic: str = "", weight: float = 1.0) -> None:
        slots = {k: v for k, v in slots.items() if not (k in _PLURAL_SLOTS and isinstance(v, int) and v == 1)}
        out.append({"cat": cat, "key": f"{cat}:{key}", "slots": slots, "topic": topic, "players": [], "weight": weight})

    for cat in ("bit_coffee", "bit_predictions", "bit_rivalry_booth", "crowd_arena", "road_life", "rink_food", "superstitions",
                "gear_talk", "old_time", "kids_hockey", "weather", "dressing_room"):
        add(cat, cat, {"city": city, "team": H, "opp": A})
    try:
        rows = {str(r.get("team_id")): r for r in session.standings.as_table_rows()}
    except Exception:
        rows = {}
    for tid, tm in teams.items():
        r = rows.get(tid)
        if r and r.get("last_10"):
            try:
                wins = int(str(r["last_10"]).split("-")[0])
            except ValueError:
                wins = 0
            if wins >= 6:
                add("form_banter", tid, {"team": _team_label(tm), "last10": str(r["last_10"]), "wins": wins}, _team_label(tm))
    year = int(getattr(session, "season_calendar_year", 0) or 0)
    hist = {tid: TEAM_HISTORY.get(_abbr(tm)) for tid, tm in teams.items()}
    for tid, tm in teams.items():
        h = hist.get(tid)
        if h and year:
            add("hist_years", tid, {"team": _team_label(tm), "years": max(1, year - h[0] + 1), "first_season": h[0]}, _team_label(tm))
        mem = getattr(tm, "memory", None)
        if mem is not None:
            apps = int(getattr(mem, "playoff_appearances", 0) or 0)
            ser = int(getattr(mem, "series_wins", 0) or 0)
            since = int(getattr(mem, "years_since_playoffs", 0) or 0)
            sl = {"team": _team_label(tm)}
            if apps >= 2:
                sl["apps"] = apps
            if ser >= 2:
                sl["series"] = ser
            if since >= 2:
                sl["since"] = since
            if len(sl) > 1:
                add("franchise_memory", tid, sl, _team_label(tm), 0.8)
    hh, ha = hist.get(str(home_id)), hist.get(str(away_id))
    if hh and ha:
        if hh[0] != ha[0]:
            old, new = (home, away) if hh[0] < ha[0] else (away, home)
            oy, ny = min(hh[0], ha[0]), max(hh[0], ha[0])
            add("hist_compare", "x", {"old_team": _team_label(old), "new_team": _team_label(new), "old_year": oy, "new_year": ny, "gap": ny - oy})
        if hh[2] + ha[2] >= 2:
            add("hist_matchup_cups", "x", {"team": H, "opp": A, "cups_a": hh[2], "cups_b": ha[2], "total": hh[2] + ha[2]})
    results = list(getattr(session, "game_results", None) or [])
    meet = [g for g in results if {str(g.get("home_id")), str(g.get("away_id"))} == {str(home_id), str(away_id)}]
    if len(meet) >= 2:
        w = sum(1 for x in meet if str(x.get("winner_id")) == str(home_id))
        add("save_meetings", str(len(meet)), {"team": H, "opp": A, "n": len(meet), "w": w, "l": len(meet) - w}, H)
    if meet:
        g = meet[-1]
        hg, ag = int(g.get("home_goals", 0) or 0), int(g.get("away_goals", 0) or 0)
        win = teams.get(str(g.get("winner_id") or (g.get("home_id") if hg > ag else g.get("away_id"))))
        if win is not None:
            add("save_last_result", str(g.get("game_id")), {"winner": _team_label(win), "score": f"{max(hg, ag)}–{min(hg, ag)}"}, _team_label(win))
    return out


def live_colour_facts(home_label: str, away_label: str, score_home: int, score_away: int, t: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if score_home > score_away:
        lead = score_home - score_away
        out.append({"cat": "crowd_live", "key": f"crowd_live:{score_home}-{score_away}", "slots": {"team": home_label, "lead_text": f"{lead}" if lead > 1 else "one"},
                    "topic": "the crowd", "players": [], "weight": 1.0})
    return out


def select2(lang: str, facts: List[Dict[str, Any]], told: Dict[str, int], game_no: int, rng: random.Random, *,
            on_ice: Optional[set] = None, used_lines: Optional[set] = None, cat_counts: Optional[Dict[str, int]] = None,
            fam_counts: Optional[Dict[str, int]] = None, last_family: Optional[str] = None, last_cat: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Family-balanced pick (target 40/30/30 sim/colour/history) with a recent-family penalty;
    never two sim stories in a row when colour or history is eligible."""
    tables = STORIES.get(lang) or STORIES["en"]
    used = used_lines if used_lines is not None else set()
    by_fam: Dict[str, List[Dict[str, Any]]] = {"sim": [], "colour": [], "history": []}
    for f in facts:
        cat = f["cat"]
        if cat not in tables or cat == last_cat:
            continue
        if cat_counts and cat_counts.get(cat, 0) >= (3 if family(cat) == "sim" else 2):
            continue
        if cat not in COMBO_KEYED:
            last = told.get(f["key"])
            if last is not None and game_no - last < 10:
                continue
        by_fam[family(cat)].append(f)
    total = sum((fam_counts or {}).values()) or 0
    scores = {}
    for fam, lst in by_fam.items():
        if not lst:
            continue
        share = ((fam_counts or {}).get(fam, 0) / total) if total else 0.0
        sc = FAMILY_TARGET[fam] - share + rng.uniform(0.0, 0.08)
        if fam == last_family:
            sc -= 0.25
        scores[fam] = sc
    if last_family == "sim" and "sim" in scores and len(scores) > 1:
        scores.pop("sim")
    for fam in sorted(scores, key=lambda k: -scores[k]):
        lst = list(by_fam[fam])
        rng.shuffle(lst)
        lst.sort(key=lambda f: -float(f.get("weight", 1.0)) * (2.0 if on_ice and any(p in on_ice for p in f.get("players") or []) else 1.0) * rng.uniform(0.6, 1.4))
        for f in lst[:14]:
            tb = tables[f["cat"]]
            combos = [(i, j) for i in range(len(tb["s"])) for j in range(len(tb["p"]))]
            rng.shuffle(combos)
            for i, j in combos:
                if ("s", f["cat"], i) in used or ("p", f["cat"], j) in used:
                    continue
                ck = f"{f['cat']}:{i}:{j}:{f['key']}"
                if f["cat"] in COMBO_KEYED:
                    last = told.get(ck)
                    if last is not None and game_no - last < 10:
                        continue
                s = _fill(tb["s"][i], f["slots"])
                p = _fill(tb["p"][j], f["slots"])
                if s is None or p is None:
                    continue
                reacts = [(k, x) for k, x in ((k, _fill(r, f["slots"])) for k, r in enumerate(tb["r"])) if x and ("r", f["cat"], k) not in used]
                rk, r = rng.choice(reacts) if reacts else (None, None)
                used.update({("s", f["cat"], i), ("p", f["cat"], j)})
                if rk is not None:
                    used.add(("r", f["cat"], rk))
                return {"key": ck if f["cat"] in COMBO_KEYED else f["key"], "cat": f["cat"], "family": fam, "tid": f"{f['cat']}:{i}:{j}",
                        "setup": s, "payoff": p, "react": r, "topic": f.get("topic") or "", "slots": dict(f["slots"])}
    return None


# ── Round 4b: paired categories (payoffs that only fit their own setup). "pp"[i] = payoffs for setup i.
EN["bit_coffee"]["pp"] = [
    ["Third cup. I'm pacing myself for overtime.", "Just saving my voice for the big moments.", "Quiet means I'm concentrating. Don't read into it."],
    ["Somebody has to keep this booth awake through the second period.", "Holds about a litre. I'm not proud of that.", "It was a gift. It would be rude not to use it."],
    ["Press box coffee. I've built up a tolerance over the years.", "The good stuff ran out in warmups. Story of my life.", "I don't think this building has ever had good coffee."],
    ["Doctor's orders. I don't want to talk about it.", "Half and half. A compromise.", "I'm told it tastes the same. It does not."],
    ["And every year I make it to about the first intermission.", "This is year eleven of cutting back.", "I've cut back to one cup at a time."],
]
EN["bit_rivalry_booth"]["pp"] = [
    ["Loser buys dinner on the next road trip. I'm feeling good about it.", "Whoever's team finishes higher picks the restaurant all of next season.", "I'm not worried. Ask me again in March."],
    ["I'm up two games to one. Nobody asked, but I'm up.", "We're dead even, which means I'm winning on style.", "Current standings: me, comfortably."],
    ["We agreed not to discuss it. Moving on.", "Let's just say both of us need more cardio.", "The scoresheet was lost. Very convenient for you."],
    ["I scored. That's all anybody needs to remember.", "I stayed on my feet the whole time. Mostly.", "There's no video, so it never happened."],
    ["Strictly theoretical. I'm a student of the game.", "Every summer. Every summer it stays exactly the same.", "In the driveway. The garage door has seen things."],
]
EN["rink_food"]["pp"] = [
    ["The loaded fries. I regret nothing.", "Some kind of brisket sandwich. I'm a fan.", "Not yet. I'm saving it for the second intermission."],
    ["Let's say the soup is ambitious.", "The sandwiches are fine. The coffee is a different story.", "Somebody brought the good cookies again, so it's a win."],
    ["A hot dog and a coffee. Classic for a reason.", "Popcorn. You can eat it with one hand and still take notes.", "A soft pretzel with mustard. I won't hear otherwise."],
    ["The line was long, but the poutine was worth it.", "I got talking to an usher who's worked here thirty years. Great stories.", "Took the long way around the concourse. You see everything that way."],
    ["Somebody's making mini doughnuts down there. Dangerous.", "That's the barbecue stand. Every visiting team asks about it.", "Fresh popcorn. The whole press box smells like it now."],
]
EN["superstitions"]["pp"] = [
    ["Same stick tape, same pregame meal, same seat on the bus. Don't touch any of it.", "Every room has its rituals, and nobody admits to half of them.", "Some guys won't step on the logo in the dressing room."],
    ["Left skate first, always. I did it in house league and I still do it.", "Same pen, same notebook, same seat up here.", "I don't wash my lucky tie during a winning streak."],
    ["One guy taps every teammate's pads in the same order before the anthem.", "A player went through the exact same stretching routine, down to the second.", "Somebody re-taped his stick three times before warmups were over."],
    ["There's always someone who refuses to shave during a winning streak.", "One guy has to be the last off the ice after warmups, every single time.", "Somebody always sits in the same stall, even on the road."],
    ["They'll tell you it's just focus. It's not just focus.", "Some of them have names for their posts. I'm not joking.", "A tap on each post before every period. Like clockwork."],
]
EN["gear_talk"]["pp"] = [
    ["Light as a feather, and they flex like a fishing rod.", "Composite everything now. A puck comes off them like a rocket.", "Players fuss over the curve like it's a work of art."],
    ["When I started, the shin pads were basically magazines and tape.", "Helmets, visors, neck guards. The safety side has improved so much.", "Even the gloves are lighter and stronger than they used to be."],
    ["They're lighter than they look, and somehow they cover more net.", "Goalies today look like they fill the whole net before the puck is dropped.", "The pads are built for the butterfly now. The old ones were for stand-up goalies."],
    ["Custom fit, heat molded. We used to wear our older brother's.", "The blades get sharpened between periods for some guys.", "Lighter boots, better support. The skating has gotten that much faster."],
    ["Heavy, and you'd break three a week. The stick boys loved us.", "You'd tape the blade until it weighed a ton.", "Every one was a little different. You'd have a favourite and guard it."],
]
EN["old_time"]["pp"] = [
    ["Shifts used to be two minutes. Now it's forty seconds and back to the bench.", "Everybody can skate now. The fourth line can fly.", "Systems everywhere now. We just chased the puck."],
    ["Maybe the old days, just to see if I could handle the hooking.", "I'd take today. Better equipment, better ice.", "Every era has its charm, but I'd want today's trainers."],
    ["Everybody can skate now. The fourth line can fly.", "The defencemen move like forwards now.", "You can't hide a slow skater in this league anymore."],
    ["And they're half right. It was tough. It was also slow.", "They'll also tell you the ice was better. It wasn't.", "Every generation says that about the one before it."],
    ["Shifts used to be two minutes. Now it's forty seconds and back to the bench.", "Coaches roll four lines now. Back then the top guys played half the game.", "Short, hard shifts. That's the modern game."],
]
EN["kids_hockey"]["pp"] = [
    ["Six in the morning, cold rink, my dad with a coffee in the stands. Every Saturday.", "Minor hockey tournaments in small towns. Hotel pools and hockey.", "Road hockey until somebody yelled car."],
    ["I was five. I spent more time on my backside than on my skates.", "Bob skates on the frozen backyard. Very glamorous.", "My first pair were two sizes too big so I could grow into them."],
    ["Six in the morning, cold rink, my dad with a coffee in the stands. Every Saturday.", "The smell of the rink at dawn. You never forget it.", "Hot chocolate from the vending machine after practice. Best part."],
    ["Somebody out there is going to be on this ice in fifteen years.", "Some kid is going to go home and practise that move in the driveway.", "Those kids in the front row will remember this game for years."],
    ["Every winter. Shovel the snow, play until your feet go numb.", "Pond hockey until it got dark, then a little longer.", "Boots for goalposts. Games went on for hours."],
]
EN["weather"]["pp"] = [
    ["Perfect hockey weather. You want it cold when you come to the rink.", "The kind of cold that makes you glad you're inside.", "It's that time of year. Everybody's got a scarf."],
    ["Windy, grey, and you couldn't ask for a better night for a game.", "Can't complain. The rink is warm and the coffee is hot.", "Wet. Let's just say wet."],
    ["If it snows, the drive home will be an adventure.", "Cold and clear, I'm told. Good night to be at the rink.", "Mixed bag. Pack for everything."],
    ["If it snows, the drive home will be an adventure.", "A little snow on a hockey night feels right.", "They'll be shovelling the parking lot when we get out."],
    ["Brisk. Let's call it brisk.", "Long enough to wake you up.", "I parked at the far end. Rookie mistake."],
]
EN["booth"]["pp"] = [
    ["Strong enough to keep me sharp through overtime, let's put it that way.", "Somebody made a fresh pot. A small miracle.", "I've had better. I've also had much worse."],
    ["He wants us to talk about the game. Revolutionary concept.", "Apparently we're over time on something. Let's keep going.", "He's giving me the thumbs up. Or he's stretching."],
    ["Somebody ordered the soup again. Brave choice.", "The cookies are gone already. Suspicious.", "It's a sandwich situation tonight."],
    ["I've had the same pen for six seasons. Not changing now.", "Lucky pen, lucky notebook. Can't be too careful.", "Lost it at the morning skate. Borrowed one. We'll see."],
]
PAIRED_CATS = frozenset(k for k, v in EN.items() if "pp" in v)

# fr/es/de/sv paired payoffs + one travel category each
FR["rink_food"]["pp"] = [["La poutine. Aucun regret.", "Pas encore, je la garde pour l'entracte."], ["Disons que la soupe est ambitieuse.", "Quelqu'un a apporté des biscuits maison. Une victoire."], ["Un hot-dog et un café. Un classique.", "Du maïs soufflé. On peut prendre des notes en même temps."]]
FR["superstitions"]["pp"] = [["Même ruban, même repas, même siège dans l'autobus.", "Certains refusent de marcher sur le logo du vestiaire."], ["Le patin gauche en premier, toujours.", "Même crayon, même cahier, même siège ici."], ["Il y en a toujours un qui refuse de se raser pendant une séquence.", "Un gars doit toujours être le dernier à quitter la glace à l'échauffement."]]
FR["old_time"]["pp"] = [["Les présences duraient deux minutes. Maintenant, quarante secondes.", "Des systèmes partout. Avant, on courait après la rondelle."], ["Tout le monde patine maintenant, même le quatrième trio.", "Les défenseurs bougent comme des attaquants."], ["Les présences duraient deux minutes. Maintenant, quarante secondes.", "Des présences courtes et intenses. C'est le hockey moderne."]]
FR["kids_hockey"]["pp"] = [["Six heures du matin, aréna froide, mon père avec un café dans les gradins.", "Du hockey de ruelle jusqu'à ce que quelqu'un crie auto."], ["Six heures du matin, aréna froide, mon père avec un café dans les gradins.", "Le chocolat chaud de la machine après la pratique."], ["Un de ces jeunes jouera sur cette glace dans quinze ans.", "Un enfant va rentrer pratiquer ce jeu dans l'entrée de garage."]]
FR["booth"]["pp"] = [["Assez fort pour me garder éveillé jusqu'en prolongation.", "Quelqu'un a fait du café frais. Un petit miracle."], ["Il veut qu'on parle du match. Quelle idée.", "Il lève le pouce. Ou il s'étire."]]
FR["road_life"] = {"s": ["Comment s'est passé le vol?", "Les voyages à ce temps-ci de l'année...", "Le calendrier a été chargé."],
                   "p": ["Deux heures de retard, puis l'autobus, puis directement à l'aréna.", "Quatre villes en six soirs. On oublie dans quel hôtel on dort.", "On apprend à dormir n'importe où."],
                   "r": ["Et il faut quand même avoir des jambes.", "D'où l'importance de la profondeur."]}
ES["rink_food"]["pp"] = [["Las patatas cargadas. Sin arrepentimientos.", "Todavía no, lo guardo para el descanso."], ["Digamos que la sopa es ambiciosa.", "Alguien trajo galletas caseras. Una victoria."], ["Un perrito caliente y un café. Un clásico.", "Palomitas. Se pueden comer tomando notas."]]
ES["superstitions"]["pp"] = [["La misma cinta, la misma comida, el mismo asiento en el autobús.", "Algunos no pisan el escudo del vestuario."], ["Siempre el patín izquierdo primero.", "El mismo bolígrafo, el mismo cuaderno, el mismo asiento aquí."], ["Siempre hay uno que no se afeita durante una racha.", "Uno siempre tiene que ser el último en salir del calentamiento."]]
ES["old_time"]["pp"] = [["Los turnos duraban dos minutos. Ahora, cuarenta segundos.", "Sistemas por todas partes. Antes perseguíamos el disco."], ["Ahora todos patinan, hasta la cuarta línea.", "Los defensas se mueven como delanteros."], ["Los turnos duraban dos minutos. Ahora, cuarenta segundos.", "Turnos cortos e intensos. Es el hockey moderno."]]
ES["kids_hockey"]["pp"] = [["Seis de la mañana, pista fría, mi padre con un café en la grada.", "Hockey en la calle hasta que alguien gritaba coche."], ["Seis de la mañana, pista fría, mi padre con un café en la grada.", "El chocolate caliente de la máquina después del entrenamiento."], ["Alguno de esos niños jugará en esta pista dentro de quince años.", "Algún niño irá a casa a practicar esa jugada."]]
ES["booth"]["pp"] = [["Lo bastante fuerte para aguantar hasta la prórroga.", "Alguien hizo café recién hecho. Un pequeño milagro."], ["Quiere que hablemos del partido. Qué ocurrencia.", "Levanta el pulgar. O se está estirando."]]
ES["road_life"] = {"s": ["¿Qué tal el vuelo?", "Los viajes en esta época del año...", "El calendario ha sido intenso."],
                   "p": ["Dos horas de retraso, luego el autobús y directo a la pista.", "Cuatro ciudades en seis noches. Ya no sabes en qué hotel estás.", "Aprendes a dormir en cualquier parte."],
                   "r": ["Y aun así hay que tener piernas.", "Por eso importa la profundidad."]}
DE["rink_food"]["pp"] = [["Die beladenen Pommes. Keine Reue.", "Noch nicht, ich hebe es mir für die Pause auf."], ["Sagen wir, die Suppe ist ehrgeizig.", "Jemand hat selbstgebackene Kekse mitgebracht. Ein Sieg."], ["Eine Bratwurst und ein Kaffee. Ein Klassiker.", "Popcorn. Damit kann man nebenbei Notizen machen."]]
DE["superstitions"]["pp"] = [["Gleiches Tape, gleiches Essen, gleicher Platz im Bus.", "Manche treten nie auf das Logo in der Kabine."], ["Immer zuerst der linke Schlittschuh.", "Gleicher Stift, gleiches Notizbuch, gleicher Platz hier oben."], ["Einer rasiert sich während einer Siegesserie nie.", "Einer muss beim Aufwärmen immer als Letzter vom Eis."]]
DE["old_time"]["pp"] = [["Früher dauerte ein Wechsel zwei Minuten. Heute vierzig Sekunden.", "Überall Systeme. Früher sind wir der Scheibe hinterhergejagt."], ["Heute können alle laufen, sogar die vierte Reihe.", "Die Verteidiger bewegen sich wie Stürmer."], ["Früher dauerte ein Wechsel zwei Minuten. Heute vierzig Sekunden.", "Kurze, harte Wechsel. Das ist modernes Eishockey."]]
DE["kids_hockey"]["pp"] = [["Sechs Uhr morgens, kalte Halle, mein Vater mit einem Kaffee auf der Tribüne.", "Straßenhockey, bis jemand Auto rief."], ["Sechs Uhr morgens, kalte Halle, mein Vater mit einem Kaffee auf der Tribüne.", "Heiße Schokolade aus dem Automaten nach dem Training."], ["Eines dieser Kinder steht in fünfzehn Jahren auf diesem Eis.", "Irgendein Kind übt diesen Spielzug morgen in der Einfahrt."]]
DE["booth"]["pp"] = [["Stark genug für eine Verlängerung, sagen wir es so.", "Jemand hat frischen Kaffee gemacht. Ein kleines Wunder."], ["Er will, dass wir über das Spiel reden. Was für eine Idee.", "Er zeigt den Daumen hoch. Oder er streckt sich."]]
DE["road_life"] = {"s": ["Wie war der Flug?", "Reisen um diese Jahreszeit...", "Der Spielplan war zuletzt brutal."],
                   "p": ["Zwei Stunden Verspätung, dann der Bus, dann direkt in die Halle.", "Vier Städte in sechs Nächten. Man weiß nicht mehr, in welchem Hotel man ist.", "Man lernt, überall zu schlafen."],
                   "r": ["Und trotzdem müssen die Beine da sein.", "Deshalb ist Tiefe so wichtig."]}
SV["rink_food"]["pp"] = [["Pommesen med allt. Ångrar ingenting.", "Inte än, jag sparar det till pausen."], ["Låt oss säga att soppan är ambitiös.", "Någon tog med hembakade kakor. En seger."], ["En korv och en kaffe. En klassiker.", "Popcorn. Man kan anteckna samtidigt."]]
SV["superstitions"]["pp"] = [["Samma tejp, samma måltid, samma plats på bussen.", "Vissa kliver aldrig på loggan i omklädningsrummet."], ["Vänster skridsko först, alltid.", "Samma penna, samma block, samma plats här uppe."], ["Det finns alltid någon som inte rakar sig under en segersvit.", "En måste alltid vara sist av isen efter uppvärmningen."]]
SV["old_time"]["pp"] = [["Förr var ett byte två minuter. Nu fyrtio sekunder.", "System överallt. Förr jagade vi bara pucken."], ["Alla kan åka skridskor nu, även fjärdekedjan.", "Backarna rör sig som forwards."], ["Förr var ett byte två minuter. Nu fyrtio sekunder.", "Korta, hårda byten. Det är modern hockey."]]
SV["kids_hockey"]["pp"] = [["Klockan sex på morgonen, kall rink, pappa med en kaffe på läktaren.", "Gatuhockey tills någon ropade bil."], ["Klockan sex på morgonen, kall rink, pappa med en kaffe på läktaren.", "Varm choklad från automaten efter träningen."], ["Ett av de där barnen spelar på den här isen om femton år.", "Något barn kommer att öva på den där finten på uppfarten."]]
SV["booth"]["pp"] = [["Starkt nog för att klara en förlängning.", "Någon har bryggt nytt kaffe. Ett litet mirakel."], ["Han vill att vi pratar om matchen. Vilken idé.", "Han visar tummen upp. Eller så sträcker han på sig."]]
SV["road_life"] = {"s": ["Hur var flyget?", "Resor den här tiden på året...", "Schemat har varit tufft."],
                   "p": ["Två timmar försenat, sen buss, sen direkt till arenan.", "Fyra städer på sex kvällar. Man vet inte vilket hotell man bor på.", "Man lär sig sova var som helst."],
                   "r": ["Och ändå ska benen vara med.", "Därför är bredden så viktig."]}
COLOUR_CATS = COLOUR_CATS | frozenset({"road_life"})
FOUNDING_CATS = frozenset({"hist_age", "hist_years", "hist_compare", "hist_drought"})
REPLIES_BY_FAMILY["en"]["colour"] = ["Ha. Classic.", "I believe it.", "Same here, honestly.", "Can't argue with that.", "That's the truth.", "Every time."]


def _pairs(tb: Dict[str, Any]) -> List[Tuple[int, int, Optional[str]]]:
    """(setup index, payoff index, paired payoff text or None)."""
    if "pp" in tb:
        return [(i, j, p) for i, lst in enumerate(tb["pp"]) for j, p in enumerate(lst) if i < len(tb["s"])]
    return [(i, j, None) for i in range(len(tb["s"])) for j in range(len(tb["p"]))]


def dataset_counts() -> Dict[str, Any]:  # noqa: F811 (round-4b: paired categories counted by real pairs)
    out: Dict[str, Any] = {}
    for lang, cats in STORIES.items():
        by = {"sim": 0, "colour": 0, "history": 0}
        lines = 0
        for k, c in cats.items():
            by[family(k)] += len(_pairs(c))
            lines += len(c["s"]) + len(c["r"]) + (sum(len(x) for x in c["pp"]) if "pp" in c else len(c["p"]))
        out[lang] = {"categories": len(cats), "composed_exchanges": sum(by.values()), "authored_lines": lines, "by_family": by}
    return out


_select2_v1 = select2


def select2(lang: str, facts: List[Dict[str, Any]], told: Dict[str, int], game_no: int, rng: random.Random, *,  # noqa: F811
            on_ice: Optional[set] = None, used_lines: Optional[set] = None, cat_counts: Optional[Dict[str, int]] = None,
            fam_counts: Optional[Dict[str, int]] = None, last_family: Optional[str] = None, last_cat: Optional[str] = None,
            told_game: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """Family-balanced pick with paired categories, once-per-game facts and a founding-year cap."""
    tables = STORIES.get(lang) or STORIES["en"]
    used = used_lines if used_lines is not None else set()
    tg = told_game if told_game is not None else set()
    founding = sum((cat_counts or {}).get(c, 0) for c in FOUNDING_CATS)
    by_fam: Dict[str, List[Dict[str, Any]]] = {"sim": [], "colour": [], "history": []}
    for f in facts:
        cat = f["cat"]
        if cat not in tables or cat == last_cat or f["key"] in tg:
            continue
        if cat in FOUNDING_CATS and founding >= globals().get("FOUNDING_CAP", 2):
            continue
        if cat_counts and cat_counts.get(cat, 0) >= (3 if family(cat) == "sim" else 2):
            continue
        if cat not in COMBO_KEYED:
            last = told.get(f["key"])
            if last is not None and game_no - last < 10:
                continue
        by_fam[family(cat)].append(f)
    total = sum((fam_counts or {}).values()) or 0
    scores = {}
    for fam, lst in by_fam.items():
        if not lst:
            continue
        share = ((fam_counts or {}).get(fam, 0) / total) if total else 0.0
        sc = FAMILY_TARGET[fam] - share + rng.uniform(0.0, 0.08)
        if fam == last_family:
            sc -= 0.25
        scores[fam] = sc
    if last_family == "sim" and "sim" in scores and len(scores) > 1:
        scores.pop("sim")
    for fam in sorted(scores, key=lambda k: -scores[k]):
        lst = list(by_fam[fam])
        rng.shuffle(lst)
        lst.sort(key=lambda f: -float(f.get("weight", 1.0)) * (2.0 if on_ice and any(p in on_ice for p in f.get("players") or []) else 1.0) * rng.uniform(0.6, 1.4))
        for f in lst[:14]:
            tb = tables[f["cat"]]
            combos = _pairs(tb)
            rng.shuffle(combos)
            for i, j, ptxt in combos:
                if ("s", f["cat"], i) in used or ("p", f["cat"], i if ptxt is not None else j, j) in used:
                    continue
                ck = f"{f['cat']}:{i}:{j}:{f['key']}"
                if f["cat"] in COMBO_KEYED:
                    last = told.get(ck)
                    if last is not None and game_no - last < 10:
                        continue
                s = _fill(tb["s"][i], f["slots"])
                p = _fill(ptxt if ptxt is not None else tb["p"][j], f["slots"])
                if s is None or p is None:
                    continue
                reacts = [(k, x) for k, x in ((k, _fill(r, f["slots"])) for k, r in enumerate(tb["r"])) if x and ("r", f["cat"], k) not in used]
                rk, r = rng.choice(reacts) if reacts else (None, None)
                used.update({("s", f["cat"], i), ("p", f["cat"], i if ptxt is not None else j, j)})
                if rk is not None:
                    used.add(("r", f["cat"], rk))
                tg.add(f["key"])
                return {"key": ck if f["cat"] in COMBO_KEYED else f["key"], "cat": f["cat"], "family": fam, "tid": f"{f['cat']}:{i}:{j}",
                        "setup": s, "payoff": p, "react": r, "topic": f.get("topic") or "", "slots": dict(f["slots"])}
    return None

EN["dressing_room"] = {
    "s": ["You ever think about what goes on in that dressing room between periods?", "Let's talk about the room for a second.",
          "People always ask me what a dressing room is really like.", "The stuff nobody sees behind those doors.", "Here's something fans don't get to see."],
    "p": ["Mostly quiet. A few words from the coach, a few from the leaders, and lots of tape.", "There's always one guy cracking jokes to keep everybody loose.",
          "The trainers are the busiest people in the building. Skates, sticks, ice packs, all of it.", "Music before the game, silence after a bad period. You learn to read the room fast.",
          "Every team has a different routine, but they all have one. Nobody skips it."],
    "r": ["That's where seasons get won.", "Culture matters more than people think.", "You can feel it when a room is together."],
}
COLOUR_CATS = COLOUR_CATS | frozenset({"dressing_room"})
COMBO_KEYED = COLOUR_CATS | HISTORY_CATS

EN["hist_one_cup"] = {
    "s": ["A little franchise history for you.", "Trophy case check.", "Some history on this club.", "Before the next faceoff, a history note.", "Banner talk for a second."],
    "p": ["{team} has one Stanley Cup to its name.", "There's exactly one championship banner for {team}.", "{team} won the Cup once, and the fans still talk about it.",
          "One Stanley Cup in franchise history for {team}.", "{team} has lifted the Cup one time."],
    "r": ["Everybody wants a second one.", "That one banner means a lot around here.", "The first one is always special."],
}
FR["hist_one_cup"] = {"s": ["Un peu d'histoire.", "Petite note sur cette concession."], "p": ["{team} a remporté la coupe Stanley une fois.", "Une seule coupe Stanley dans l'histoire de {team}.", "{team} a soulevé la coupe une fois."], "r": ["Tout le monde en veut une deuxième.", "Cette bannière compte beaucoup ici."]}
ES["hist_one_cup"] = {"s": ["Un poco de historia.", "Una nota sobre esta franquicia."], "p": ["{team} ha ganado la Copa Stanley una vez.", "Una sola Copa Stanley en la historia de {team}.", "{team} levantó la Copa una vez."], "r": ["Todos quieren la segunda.", "Ese estandarte significa mucho aquí."]}
DE["hist_one_cup"] = {"s": ["Ein bisschen Geschichte.", "Eine Notiz zu dieser Franchise."], "p": ["{team} hat den Stanley Cup einmal gewonnen.", "Ein einziger Stanley Cup in der Geschichte von {team}.", "{team} hat den Cup einmal gestemmt."], "r": ["Alle wollen einen zweiten.", "Dieses Banner bedeutet hier viel."]}
SV["hist_one_cup"] = {"s": ["Lite historia.", "En notis om klubben."], "p": ["{team} har vunnit Stanley Cup en gång.", "En enda Stanley Cup i {team}s historia.", "{team} har lyft bucklan en gång."], "r": ["Alla vill ha en till.", "Den banderollen betyder mycket här."]}
HISTORY_CATS = HISTORY_CATS | frozenset({"hist_one_cup"})
COMBO_KEYED = COLOUR_CATS | HISTORY_CATS
FOUNDING_CAP = 3

EN["save_big_win"] = {
    "s": ["Looking back at {team}'s season so far.", "Here's one from the {team} highlight reel this year.", "Best night of the year for {team}?",
          "Something from earlier this season.", "Let's rewind a few weeks for {team}."],
    "p": ["The biggest win so far was {score} over {opp}.", "{team} beat {opp} {score}. Most lopsided result of their season.",
          "That {score} win against {opp} still stands out.", "Their high-water mark: {score} over {opp}.", "A {score} win over {opp}, the biggest margin they've had this year."],
    "r": ["Nights like that build confidence.", "Everything went in that night.", "They'd love another one of those."],
}
EN["save_win_streak"] = {
    "s": ["How hot has {team} gotten this season?", "Streak talk from earlier this year.", "Here's a note from the {team} season so far.",
          "Remember when {team} couldn't lose?", "A little season history on {team}."],
    "p": ["Their longest winning streak this season is {n} games.", "{team} put together {n} straight wins at one point.",
          "The best run so far: {n} wins in a row.", "{n} consecutive wins is the season high for {team}.", "At their best, {team} won {n} on the trot."],
    "r": ["That's the version of this team they want back.", "Streaks like that change a season.", "Fun stretch for the fans."],
}
for _L, _bw, _ws in (
    ("fr", {"s": ["Retour sur la saison de {team}.", "Le meilleur soir de l'année pour {team}?"], "p": ["La plus grosse victoire : {score} contre {opp}.", "{team} a battu {opp} {score}.", "Cette victoire de {score} contre {opp} ressort encore."], "r": ["Ces soirs-là donnent confiance.", "Tout rentrait ce soir-là."]},
            {"s": ["Jusqu'où {team} s'est-il emballé cette saison?", "Un peu d'histoire de la saison de {team}."], "p": ["Sa plus longue séquence de victoires cette saison : {n} matchs.", "{team} a enchaîné {n} victoires.", "{n} victoires de suite, c'est le sommet de la saison."], "r": ["C'est cette équipe-là qu'on veut revoir.", "Ce genre de séquence change une saison."]}),
    ("es", {"s": ["Repasemos la temporada de {team}.", "¿La mejor noche del año para {team}?"], "p": ["La mayor victoria: {score} ante {opp}.", "{team} venció a {opp} por {score}.", "Esa victoria por {score} ante {opp} todavía destaca."], "r": ["Esas noches dan confianza.", "Aquella noche entraba todo."]},
            {"s": ["¿Cuánto se ha encendido {team} esta temporada?", "Un poco de historia de la temporada de {team}."], "p": ["Su racha más larga de victorias esta temporada: {n} partidos.", "{team} encadenó {n} victorias.", "{n} victorias seguidas es su mejor marca del año."], "r": ["Esa es la versión que quieren recuperar.", "Rachas así cambian una temporada."]}),
    ("de", {"s": ["Ein Blick zurück auf die Saison von {team}.", "Der beste Abend des Jahres für {team}?"], "p": ["Der höchste Sieg bisher: {score} gegen {opp}.", "{team} schlug {opp} mit {score}.", "Dieser {score}-Sieg gegen {opp} sticht noch heraus."], "r": ["Solche Abende geben Selbstvertrauen.", "An dem Abend ging alles rein."]},
            {"s": ["Wie heiß lief {team} in dieser Saison?", "Ein bisschen Saisongeschichte zu {team}."], "p": ["Die längste Siegesserie der Saison: {n} Spiele.", "{team} hat {n} Siege in Folge geschafft.", "{n} Siege am Stück sind der Saisonbestwert."], "r": ["Diese Version wollen sie zurück.", "Solche Serien prägen eine Saison."]}),
    ("sv", {"s": ["En tillbakablick på säsongen för {team}.", "Årets bästa kväll för {team}?"], "p": ["Största segern hittills: {score} mot {opp}.", "{team} slog {opp} med {score}.", "Den där segern med {score} mot {opp} sticker fortfarande ut."], "r": ["Sådana kvällar ger självförtroende.", "Allt gick in den kvällen."]},
            {"s": ["Hur het har {team} varit i år?", "Lite säsongshistoria om {team}."], "p": ["Längsta segersviten i år: {n} matcher.", "{team} vann {n} raka matcher.", "{n} raka segrar är säsongsbästa."], "r": ["Den versionen vill de ha tillbaka.", "Sådana sviter förändrar en säsong."]}),
):
    STORIES[_L]["save_big_win"] = _bw
    STORIES[_L]["save_win_streak"] = _ws
HISTORY_CATS = HISTORY_CATS | frozenset({"save_big_win", "save_win_streak"})
COMBO_KEYED = COLOUR_CATS | HISTORY_CATS

_extra_facts_v1 = extra_facts


def extra_facts(session: Any, home: Any, away: Any, home_id: str, away_id: str) -> List[Dict[str, Any]]:  # noqa: F811
    out = _extra_facts_v1(session, home, away, home_id, away_id)
    results = list(getattr(session, "game_results", None) or [])
    tbid = getattr(session, "team_by_id", {}) or {}
    for tid, tm in ((str(home_id), home), (str(away_id), away)):
        games = [g for g in results if str(g.get("home_id")) == tid or str(g.get("away_id")) == tid]
        if len(games) < 5:
            continue
        best = None
        streak = run = 0
        for g in games:
            hg, ag = int(g.get("home_goals", 0) or 0), int(g.get("away_goals", 0) or 0)
            mine, theirs = (hg, ag) if str(g.get("home_id")) == tid else (ag, hg)
            won = str(g.get("winner_id")) == tid if g.get("winner_id") else mine > theirs
            run = run + 1 if won else 0
            streak = max(streak, run)
            if won and (best is None or mine - theirs > best[0]):
                opp_id = str(g.get("away_id") if str(g.get("home_id")) == tid else g.get("home_id"))
                best = (mine - theirs, mine, theirs, opp_id, g.get("game_id"))
        T = _team_label(tm)
        if best and best[0] >= 2 and tbid.get(best[3]) is not None:
            out.append({"cat": "save_big_win", "key": f"save_big_win:{tid}:{best[4]}", "slots": {"team": T, "opp": _team_label(tbid[best[3]]), "score": f"{best[1]}–{best[2]}"},
                        "topic": T, "players": [], "weight": 1.0})
        if streak >= 3:
            out.append({"cat": "save_win_streak", "key": f"save_win_streak:{tid}:{streak}", "slots": {"team": T, "n": streak}, "topic": T, "players": [], "weight": 1.0})
    return out
