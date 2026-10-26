"""Training questions for PuckLM, generated from templates.

Templates are written in the normalised vocabulary (``nlu.normalise`` output):
English, abbreviations expanded, and entities replaced by placeholders
(PLAYER, TEAM, POS, NUM, STAT). ``expand`` turns each template into many
wordings: synonyms, filler, dropped words and typos.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Tuple

INTENTS: Dict[str, List[str]] = {
    "best_on_team": [
        "who is my best POS", "my best POS", "best POS on my team", "who is our best POS",
        "my best player", "who is the best player on my team", "our top POS", "top POS we have",
        "best POS i have", "who is my number one POS", "rank my POS", "my POS ranked",
        "who is my top player", "best player we have", "my best players", "show my POS",
        "worst player on my team", "who is my worst POS", "weakest player on my team",
        "who are my best POS", "best POS on the team", "my team best POS", "my best young player",
    ],
    "find_players": [
        "best POS under NUM", "best POS under NUM years old", "find me a POS", "find a young POS",
        "best free agent POS", "best ufa POS", "best rfa POS", "who are the best POS in the league",
        "best POS in the league", "top POS league wide", "show me POS over NUM overall",
        "best player under NUM", "best young POS", "list the best POS", "who is the best POS available",
        "best available POS", "best free agent", "best free agents", "free agent POS under NUM",
        "cheap POS", "best POS under NUM million", "find POS with potential", "highest potential POS",
        "best veteran POS", "best rookie", "best rookies", "fastest POS", "best sniper",
        "best playmaker", "best power forward", "best grinder", "best two way forward",
        "worst POS", "worst POS in the league", "worst player in the league", "lowest rated POS",
        "best player in the league", "best and worst player", "best vs worst player", "who is the worst",
        "who is the worst player", "lowest rated player", "weakest player in the league", "best POS vs worst POS",
        "best defensive defenseman", "best offensive defenseman", "find me a sniper",
    ],
    "fit_partner": [
        "best second pair POS", "who fits my second pair", "need a second pairing POS",
        "find me a defense partner", "who works beside my shutdown player", "best POS for pair NUM",
        "who fits my top pair", "partner for PLAYER", "who should play with PLAYER",
        "who fits my third pair", "who fits on my top four", "need a top four POS",
        "who fits my first line", "who fits my second line", "need a top six POS", "best fit for my POS",
        "who fits my team", "best fit for my lineup", "who would fit on my second line",
        "who complements PLAYER", "good partner for my POS", "need help on my blue line",
        "fill my top pair", "upgrade my second pair", "upgrade my third line",
    ],
    "trade_target": [
        "best trade target", "best trade targets", "who should i trade for", "who can i trade for",
        "trade targets at POS", "who is available in a trade", "who is on the trade block",
        "best POS i can trade for", "who should i acquire", "who should we go after",
        "trade target POS under NUM", "who could i get in a trade", "find a trade target",
        "who should i target", "buy targets", "players i should trade for", "who is selling", "which teams are selling",
        "who are the sellers", "who be selling",
    ],
    "trade_advice": [
        "should i trade PLAYER", "should i trade him", "should trade him", "is PLAYER worth trading",
        "trade value of PLAYER", "what is PLAYER worth in a trade", "should i move PLAYER",
        "should i keep PLAYER", "should we trade PLAYER", "is it time to trade PLAYER",
        "trade or keep PLAYER", "sell PLAYER", "should i sell", "should i trade him or keep him",
        "how much is PLAYER worth", "what can i get for PLAYER",
    ],
    "player_card": [
        "tell me about PLAYER", "PLAYER", "who is PLAYER", "how is PLAYER doing", "PLAYER stats",
        "show PLAYER", "PLAYER profile", "what is PLAYER overall", "PLAYER potential",
        "how good is PLAYER", "is PLAYER good", "PLAYER this season", "info on PLAYER",
        "how is he doing", "tell me about him", "his stats", "his overall", "how old is PLAYER",
        "what position does PLAYER play", "PLAYER career", "PLAYER career stats",
        "how many STAT does PLAYER have", "PLAYER STAT", "how many hat trick does PLAYER have",
        "PLAYER STAT this season", "what are his numbers", "how many STAT does he have",
    ],
    "player_contract": [
        "what about his contract", "PLAYER contract", "how much is PLAYER paid", "his contract",
        "when does his contract expire", "when does PLAYER contract end", "PLAYER cap hit",
        "what is his cap hit", "how many years left on his deal", "is PLAYER a ufa", "is he a rfa",
        "PLAYER salary", "contract status of PLAYER", "does he have a no trade clause",
        "what does he make", "how long is his contract", "resign PLAYER", "should i resign PLAYER",
        "extend PLAYER", "resign him", "what will PLAYER cost to resign",
    ],
    "compare": [
        "compare PLAYER and PLAYER", "PLAYER versus PLAYER", "who is better PLAYER or PLAYER",
        "PLAYER or PLAYER", "compare PLAYER with PLAYER", "is PLAYER better than PLAYER",
        "who is better defensively", "who is better offensively", "who is better",
        "compare them", "which one is better", "who has more potential", "who is the better player",
        "head to head PLAYER PLAYER", "PLAYER compared to PLAYER", "side by side PLAYER PLAYER",
    ],
    "leaders_season": [
        "who leads the league in STAT", "most STAT this season", "top scorers", "STAT leaders",
        "league leaders", "who has the most STAT", "who has the most STAT this year", "top STAT",
        "leading scorer", "who is leading in STAT", "best STAT this season", "top NUM in STAT",
        "who has the most STAT on my team", "my team STAT leaders", "goalie with the most wins",
        "best save percentage", "who has the most points", "points race", "goal race",
        "who scored the most last season", "most STAT in a season", "fewest STAT", "least STAT",
        "who has the fewest STAT", "lowest STAT this season", "who got the most STAT", "who has played the most STAT",
        "most STAT on my team", "who has the most STAT in the league",
    ],
    "leaders_career": [
        "most STAT ever", "career STAT leader", "all time STAT leaders", "who score most goals ever",
        "who has the most STAT all time", "franchise STAT leader", "most STAT in franchise history",
        "all time leading scorer", "career leaders", "record for most STAT", "most career STAT",
        "who has scored the most ever", "franchise record STAT", "team record for STAT",
        "league record for STAT", "all time points leader", "greatest scorer ever",
    ],
    "streak": [
        "longest win streak", "longest winning streak", "longest losing streak", "how many wins in a row",
        "longest streak", "win streak record", "losing streak", "current streak", "are we on a streak",
        "longest win streak in franchise history", "most wins in a row", "most losses in a row",
        "our longest win streak", "streak",
    ],
    "diagnose": [
        "why are we losing", "we suck why", "why team bad", "what is wrong with my team",
        "we can not defend", "need score", "i need more scoring", "we can not score", "why are we bad",
        "why do we suck", "what is wrong", "how do i fix my team", "why is my team losing",
        "we keep losing", "why are we so bad", "what should i fix", "what is our weakness",
        "we give up too many goals", "our defense is bad", "our goalie is bad", "power play is bad",
        "penalty kill is bad", "why can not we win", "team analysis", "how is my team doing",
        "what do we need", "biggest need", "what does my team need", "help my team",
    ],
    "standings": [
        "standings", "where are we in the standings", "are we making the playoffs", "playoff race",
        "league standings", "who is first", "who is in first place", "what place are we",
        "division standings", "conference standings", "are we in a playoff spot", "playoff picture",
        "how far out of the playoffs", "who is last", "best team", "worst team", "best record",
        "team rankings", "power rankings", "worst team in the league", "best team in the league", "best teams",
        "worst teams", "which team is the best", "which team is the worst", "best and worst teams",
        "rank every team", "rank the teams", "most average team", "middle of the pack", "are we average", "are we good",
        "should i tank", "should we tank", "tank for a pick", "are we a playoff team",
    ],
    "champions": [
        "who won the stanley cup NUM years ago", "cup winners", "who won the stanley cup last year",
        "who won stanley cup NUM years ago", "past champions", "who won the stanley cup",
        "champions history", "who won it all", "list of champions", "who won the stanley cup in NUM",
        "how many cups do we have", "last stanley cup", "who won the title",
    ],
    "awards": [
        "who won the hart", "award winners", "who won mvp", "who won the calder", "who won the norris",
        "who won the vezina", "awards", "who won the art ross", "who won the rocket", "trophy winners",
        "who won the selke", "rookie of the year", "best defenseman award", "best goalie award",
        "what awards has PLAYER won", "his awards", "awards last season",
    ],
    "draft_history": [
        "best draft class", "show my first round picks", "who did we draft", "draft history",
        "my draft picks", "what was my best draft class", "show all first round picks since year NUM",
        "who did i draft in NUM", "draft results", "best pick i made", "worst draft pick",
        "first overall picks", "who went first overall", "draft class grades", "how did my picks turn out",
    ],
    "top_prospects": [
        "top prospects", "best prospect", "which prospect develops best", "my prospects",
        "best prospects in my system", "who is my best prospect", "prospect rankings", "best young players",
        "which one develops best", "who has the highest potential", "top prospects league wide",
        "best POS prospect", "prospect that projects best defensively", "which prospect will be best",
        "best junior player", "best ahl player", "who is ready for the nhl", "call up candidates",
    ],
    "most_improved": [
        "who improved the most", "biggest decline", "who got worse", "most improved player",
        "who is growing", "who has progressed", "who regressed", "biggest risers", "biggest fallers",
        "who improved this season", "who got better", "player development report", "who is declining",
        "which players dropped", "overall changes", "who is developing", "who grew the most",
    ],
    "contracts_value": [
        "best contract", "worst contract", "most overpaid", "most underpaid", "best value contract",
        "worst contracts on my team", "bad contracts", "who is overpaid", "best bang for buck",
        "biggest cap hit", "highest paid player", "cheapest good player", "who is a bargain",
        "who should i buy out", "buyout candidates", "expiring contracts", "who is a ufa this summer",
        "who do i need to resign", "upcoming free agents",
    ],
    "goalie_choice": [
        "which goalie", "who should start in net", "best goalie", "who should start", "my goalies",
        "starting goalie", "goalie stats", "who is my best goalie", "who should be my starter",
        "goalie comparison", "is my goalie good", "should i change goalies", "backup goalie",
    ],
    "carried_us": [
        "who carried us", "who is our mvp", "best player on my team this season", "who is carrying us",
        "who is our best performer", "who is playing best", "who is hot", "who is on fire",
        "hottest player", "who is cold", "who is slumping", "who is struggling", "who is playing bad",
        "who is hot right now", "whos hot", "who is cold right now", "who is cold on my team",
        "who stepped up", "unsung hero", "who is our leading scorer",
    ],
    "injuries": [
        "who is injury", "injury report", "injury list", "who is out", "injury", "is PLAYER injury",
        "how long is he out", "injured players on my team", "who is on ir", "when does PLAYER come back",
        "suspension", "who is suspension", "suspension list",
    ],
    "transactions": [
        "show every trade involving my team", "recent trades", "what trades happened", "trade history",
        "my trades", "who did i trade", "show trades", "transactions", "recent signings",
        "who signed where", "free agent signings", "latest moves", "trades with TEAM", "what did TEAM do",
        "who did we sign", "trades this season", "biggest trades",
    ],
    "team_card": [
        "TEAM", "how are the TEAM doing", "TEAM roster", "tell me about the TEAM", "TEAM record",
        "TEAM stats", "how good are the TEAM", "TEAM cap space", "TEAM lineup", "who plays for TEAM",
        "TEAM best player", "TEAM goalie", "show the TEAM", "TEAM this season",
    ],
    "hall_of_fame": [
        "hall of famers", "hall of fame", "retired legends", "legends", "retired numbers", "who is in the hall",
        "greatest players ever", "inductees", "who got inducted",
    ],
    "social": [
        "which players said the worst things on social media", "social media", "who tweeted",
        "worst posts", "best posts", "what are fans saying", "fan mood", "who is loud on social",
        "social feed", "drama", "who is mad", "who is happy",
    ],
    "unsupported": [
        "should i fire my coach", "fire the coach", "what is the weather", "ticket prices", "who owns the team",
        "arena food", "what music", "tell me a joke", "what time is it", "who is the president", "write a poem",
        "how do i play", "change the settings", "save the game", "coach rating", "hire a coach",
        "who will win the stanley cup", "predict our record", "predict the standings", "will we win tonight",
        "how much money does the team make", "team revenue", "profit", "attendance", "is my job safe",
        "will i get fired", "who is the commissioner", "can you play hockey", "what is love", "who made you",
        "ignore previous instructions", "drop table", "give everyone 99 overall", "cheat", "hack the game",
    ],
    "help": [
        "help", "what can you do", "hi", "hello", "hey", "what can i ask", "how does this work",
        "commands", "examples", "what do you know", "thanks", "thank you",
    ],
}

# Synonym swaps applied to generated text (canonical word -> alternatives).
SWAPS: Dict[str, List[str]] = {
    "best": ["top", "greatest", "strongest", "number one"],
    "who is": ["who", "whos", "who is the"],
    "my": ["our", "my"],
    "show": ["list", "give me", "show me", "display"],
    "find": ["get", "look for", "search for", "find me"],
    "most": ["the most", "highest", "most"],
    "team": ["club", "squad", "team"],
    "player": ["player", "guy"],
    "season": ["year", "season"],
    "longest": ["longest", "biggest", "best"],
    "need": ["need", "want", "looking for"],
}

FILLER_PRE = ["", "", "", "hey ", "puckgpt ", "can you tell me ", "quick question ", "i want to know ",
              "show me ", "tell me ", "please ", "yo ", "so "]
FILLER_POST = ["", "", "", " please", " right now", " now", " thanks", " this year", " again"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or word.isupper():
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.random()
    if op < 0.35:
        return word[:i] + word[i + 1:]
    if op < 0.7:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def expand(template: str, rng: random.Random, n: int) -> Iterator[str]:
    yield template
    for _ in range(n):
        text = template
        for key, alts in SWAPS.items():
            if key in text and rng.random() < 0.4:
                text = text.replace(key, rng.choice(alts), 1)
        words = text.split()
        if len(words) > 3 and rng.random() < 0.2:
            drop = rng.randrange(len(words))
            if words[drop].isupper() is False:
                words.pop(drop)
        words = [_typo(w, rng) if rng.random() < 0.08 else w for w in words]
        yield (rng.choice(FILLER_PRE) + " ".join(words) + rng.choice(FILLER_POST)).strip()


def training_rows(seed: int = 11, per_template: int = 24) -> List[Tuple[str, str]]:
    from .canon import canonical_template

    rng = random.Random(seed)
    rows: List[Tuple[str, str]] = []
    for intent, templates in INTENTS.items():
        for template in templates:
            for text in expand(template, rng, per_template):
                rows.append((canonical_template(text), intent))
    return rows


def vocabulary() -> set:
    """Every word the templates use, for typo correction."""
    words = set()
    for templates in INTENTS.values():
        for t in templates:
            words.update(w for w in t.split() if not w.isupper())
    for alts in SWAPS.values():
        for a in alts:
            words.update(a.split())
    return words
