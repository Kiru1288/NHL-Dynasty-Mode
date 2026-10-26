"""Word lists PuckLM normalises with: hockey shorthand, slang, and seven languages.

Everything maps onto one small English vocabulary so the intent model only has
to learn English. Keys are lowercase and accent-stripped (see ``nlu.fold``).
"""

from __future__ import annotations

from typing import Dict, List

# Shorthand and slang -> canonical words. Multi-word values are fine.
ABBREVIATIONS: Dict[str, str] = {
    "ld": "left defenseman", "lhd": "left defenseman",
    "rd": "right defenseman", "rhd": "right defenseman",
    "lw": "left wing", "rw": "right wing",
    "c": "center", "centre": "center", "centers": "center", "centres": "center",
    "g": "goalie", "goalies": "goalie", "goaltender": "goalie", "goaltenders": "goalie",
    "tender": "goalie", "netminder": "goalie", "keeper": "goalie", "goalkeeper": "goalie",
    "d": "defenseman", "dman": "defenseman", "dmen": "defenseman", "d-man": "defenseman",
    "dmans": "defenseman", "defenceman": "defenseman", "defencemen": "defenseman",
    "defensemen": "defenseman", "blueliner": "defenseman", "blueliners": "defenseman",
    "dfd": "defensive defenseman", "ofd": "offensive defenseman", "twd": "two way defenseman",
    "twf": "two way forward", "ply": "playmaker", "snp": "sniper", "pwf": "power forward",
    "grn": "grinder", "f": "forward", "fwd": "forward", "forwards": "forward",
    "winger": "wing", "wingers": "wing",
    "ovr": "overall", "ovrl": "overall",
    "pot": "potential", "ceiling": "potential", "upside": "potential",
    "ahl": "ahl", "nhl": "nhl", "echl": "echl",
    "rfa": "rfa", "ufa": "ufa", "fa": "free agent", "fas": "free agent", "elc": "entry contract",
    "top6": "top six", "bot6": "bottom six", "bottom6": "bottom six",
    "top4": "top four", "top2": "top pair", "1st": "first", "2nd": "second", "3rd": "third",
    "4th": "fourth", "pp": "power play", "pk": "penalty kill", "ppg": "power play goals",
    "gaa": "goals against average", "sv": "save", "sv%": "save percentage", "svpct": "save percentage",
    "pts": "points", "pt": "points", "g/a": "goals assists", "plusminus": "plus minus", "+/-": "plus minus",
    "pim": "penalty minutes", "pims": "penalty minutes", "toi": "ice time", "gp": "games played",
    "ot": "overtime", "shutouts": "shutout",
    "hatty": "hat trick", "hattrick": "hat trick", "hatties": "hat trick",
    "w": "wins", "l": "losses", "gm": "general manager", "aav": "cap hit", "caphit": "cap hit",
    "yrs": "years", "yr": "year", "szn": "season", "sznz": "season", "seasn": "season",
    "sho": "show", "dos": "does", "owr": "our", "hwo": "how", "teh": "the", "whta": "what", "taht": "that",
    "becuase": "because", "cuz": "because", "thru": "through", "tho": "though", "abt": "about", "bout": "about",
    "da": "the", "n": "and", "nah": "", "be": "", "ya": "you", "yall": "you", "ppl": "people",
    "cookin": "hot", "cooking": "hot", "heater": "hot", "rolling": "hot", "rakin": "hot",
    "sellin": "selling", "sellers": "selling", "seller": "selling",
    "stinks": "bad", "stink": "bad", "ass": "bad", "mid": "average", "meh": "average", "sieve": "bad goalie",
    "slumpin": "cold", "slump": "cold", "slumps": "cold", "icy": "cold",
    "faceoff": "faceoffs", "draws": "faceoffs", "dot": "faceoffs",
    "u": "you", "ur": "your", "r": "are", "y": "why", "wat": "what", "wut": "what", "wht": "what",
    "whos": "who is", "who's": "who is", "whats": "what is", "what's": "what is",
    "hows": "how is", "how's": "how is", "im": "i am", "i'm": "i am", "dont": "do not",
    "don't": "do not", "cant": "can not", "can't": "can not", "wont": "will not", "won't": "will not",
    "isnt": "is not", "isn't": "is not", "aint": "is not", "gonna": "going to", "wanna": "want to",
    "gotta": "have to", "pls": "please", "plz": "please", "thx": "thanks", "rn": "right now",
    "atm": "right now", "tbh": "", "lol": "", "lmao": "", "bruh": "", "bro": "", "ngl": "", "fr": "",
    "vs": "versus", "v": "versus", "vs.": "versus", "w/": "with", "w/o": "without",
    "ppl": "people", "b4": "before", "2day": "today", "tmrw": "tomorrow", "bc": "because", "cuz": "because",
    "suk": "suck", "sux": "suck", "sucks": "suck", "sucking": "suck", "trash": "bad", "garbage": "bad",
    "awful": "bad", "terrible": "bad", "horrible": "bad", "struggling": "cold", "slumping": "cold",
    "bum": "bad", "goat": "best", "elite": "best", "stud": "best", "dawg": "best",
    "lit": "scored", "snipes": "goals", "snipe": "goal", "tallies": "goals", "tally": "goal",
    "markers": "goals", "buried": "scored", "biscuits": "goals", "apples": "assists", "helpers": "assists",
    "helper": "assist", "dimes": "assists", "dime": "assist", "twig": "stick", "barn": "arena",
    "cup": "stanley cup", "chirp": "trash talk", "tilly": "fight", "scrap": "fight", "tape": "",
    "sauce": "pass", "dangle": "deke", "mitts": "hands", "wheels": "speed", "motor": "effort",
    "kid": "prospect", "kids": "prospect", "youngster": "young", "youngsters": "young", "vet": "veteran",
    "vets": "veteran", "oldie": "veteran", "rook": "rookie", "rookies": "rookie", "prospects": "prospect",
    "deal": "trade", "dealt": "traded", "flip": "trade", "ship": "trade", "shipped": "traded",
    "re-sign": "resign", "resigning": "resign", "extend": "resign", "extension": "resign",
    "lottery": "lottery", "picks": "pick", "drafted": "draft", "drafts": "draft",
    "injured": "injury", "injuries": "injury", "hurt": "injury", "ir": "injury", "ltir": "injury",
    "suspended": "suspension", "suspensions": "suspension", "banned": "suspension",
    "dude": "player", "guy": "player", "guys": "player", "players": "player", "skater": "player",
    "skaters": "player", "boys": "team", "squad": "team", "club": "team", "lineup": "roster",
}

# The user's own team, as typed.
SELF_TEAM = {"my", "mine", "our", "ours", "us", "we", "myteam", "ourteam"}

# Words that point back at what the conversation just showed.
PRONOUNS_ONE = {"he", "him", "his", "himself", "that guy", "this guy", "that player", "this player", "dude", "the guy"}
PRONOUNS_LIST = {"them", "they", "those", "these", "which one", "one of them", "of them", "the list", "which of"}

# Positions after expansion.
POSITION_WORDS: Dict[str, str] = {
    "left defenseman": "LD", "right defenseman": "RD", "defenseman": "D", "defense": "D",
    "defence": "D", "left wing": "LW", "right wing": "RW", "wing": "W", "center": "C",
    "goalie": "G", "forward": "F",
}

ROLE_WORDS: Dict[str, str] = {
    "defensive defenseman": "DFD", "offensive defenseman": "OFD", "two way defenseman": "TWD",
    "puck moving": "OFD", "puck mover": "OFD", "net front": "PWF", "net front presence": "PWF", "crease": "PWF",
    "two way forward": "TWF", "playmaker": "PLY", "sniper": "SNP", "power forward": "PWF",
    "grinder": "GRN", "shutdown": "DFD", "enforcer": "GRN",
}

DEPTH_WORDS: Dict[str, str] = {
    "top pair": "pair1", "first pair": "pair1", "1 pair": "pair1", "pair 1": "pair1",
    "second pair": "pair2", "2 pair": "pair2", "pair 2": "pair2", "second pairing": "pair2",
    "third pair": "pair3", "bottom pair": "pair3", "pair 3": "pair3",
    "top four": "top4", "top six": "top6", "bottom six": "bot6",
    "first line": "line1", "second line": "line2", "third line": "line3", "fourth line": "line4",
}

# Stat words -> ledger keys.
STAT_WORDS: Dict[str, str] = {
    "goals": "g", "goal": "g", "scored": "g", "scorer": "pts", "scoring": "pts",
    "assists": "a", "assist": "a", "points": "pts", "point": "pts", "production": "pts",
    "plus minus": "plus_minus", "penalty minutes": "pim", "hits": "hit", "hit": "hit",
    "blocks": "blk", "blocked": "blk", "shots": "sog", "shot": "sog", "wins": "w",
    "shutout": "so", "save percentage": "sv_pct", "goals against average": "gaa",
    "games played": "gp", "games": "gp", "ice time": "toi", "power play goals": "ppg", "faceoffs": "fo_pct",
}

# Words in other languages -> English. Accents are already folded away.
FOREIGN: Dict[str, Dict[str, str]] = {
    "fr": {
        "qui": "who", "est": "is", "mon": "my", "ma": "my", "mes": "my", "notre": "our", "nos": "our",
        "meilleur": "best", "meilleure": "best", "meilleurs": "best", "pire": "worst", "joueur": "player",
        "joueurs": "player", "defenseur": "defenseman", "defenseurs": "defenseman", "gardien": "goalie",
        "attaquant": "forward", "attaquants": "forward", "centre": "center", "ailier": "wing",
        "buts": "goals", "but": "goal", "passes": "assists", "points": "points", "equipe": "team",
        "pourquoi": "why", "perdons": "losing", "perd": "losing", "perdre": "losing", "gagne": "won",
        "serie": "streak", "plus": "most", "longue": "longest", "victoires": "wins", "echange": "trade",
        "echanger": "trade", "contrat": "contract", "repechage": "draft", "espoir": "prospect",
        "espoirs": "prospect", "jeune": "young", "jeunes": "young", "ans": "years", "il": "", "y": "",
        "a": "", "de": "", "du": "", "le": "", "la": "", "les": "", "des": "", "en": "", "sur": "",
        "quel": "which", "quelle": "which", "saison": "season", "coupe": "stanley cup", "blesse": "injury",
        "compare": "compare", "comparer": "compare", "et": "and", "moins": "under", "que": "", "ce": "", "ligue": "league",
        "pire": "worst", "gagne": "won",
    },
    "es": {
        "quien": "who", "es": "is", "mi": "my", "mis": "my", "nuestro": "our", "nuestra": "our",
        "mejor": "best", "mejores": "best", "peor": "worst", "jugador": "player", "jugadores": "player",
        "defensor": "defenseman", "defensa": "defenseman", "defensores": "defenseman", "portero": "goalie",
        "arquero": "goalie", "delantero": "forward", "centro": "center", "ala": "wing", "goles": "goals",
        "gol": "goal", "asistencias": "assists", "puntos": "points", "equipo": "team", "por": "",
        "que": "", "porque": "why", "perdemos": "losing", "perdiendo": "losing", "racha": "streak",
        "mas": "most", "larga": "longest", "victorias": "wins", "intercambio": "trade", "cambiar": "trade",
        "contrato": "contract", "draft": "draft", "prospecto": "prospect", "joven": "young", "jovenes": "young",
        "anos": "years", "hace": "ago", "gano": "won", "copa": "stanley cup", "temporada": "season",
        "cual": "which", "el": "", "la": "", "los": "", "las": "", "de": "", "del": "", "en": "",
        "lesionado": "injury", "comparar": "compare", "y": "and", "menos": "under", "liga": "league", "peor": "worst",
    },
    "de": {
        "wer": "who", "ist": "is", "mein": "my", "meine": "my", "meiner": "my", "unser": "our",
        "unsere": "our", "bester": "best", "beste": "best", "besten": "best", "schlechteste": "worst",
        "spieler": "player", "verteidiger": "defenseman", "torwart": "goalie", "torhuter": "goalie",
        "sturmer": "forward", "mittelsturmer": "center", "flugel": "wing", "tore": "goals", "tor": "goal",
        "vorlagen": "assists", "punkte": "points", "mannschaft": "team", "team": "team", "warum": "why",
        "verlieren": "losing", "verlieren wir": "losing", "serie": "streak", "siegesserie": "win streak",
        "langste": "longest", "meisten": "most", "siege": "wins", "tausch": "trade", "tauschen": "trade",
        "vertrag": "contract", "draft": "draft", "talent": "prospect", "jung": "young", "junge": "young",
        "jahre": "years", "vor": "ago", "gewann": "won", "pokal": "stanley cup", "saison": "season",
        "welcher": "which", "der": "", "die": "", "das": "", "den": "", "dem": "", "wir": "we",
        "verletzt": "injury", "vergleiche": "compare", "und": "and", "unter": "under", "liga": "league", "hat": "has",
    },
    "sv": {
        "vem": "who", "ar": "is", "min": "my", "mitt": "my", "mina": "my", "var": "our", "vart": "our",
        "basta": "best", "bast": "best", "samsta": "worst", "spelare": "player", "back": "defenseman",
        "backar": "defenseman", "malvakt": "goalie", "forward": "forward", "center": "center",
        "mal": "goals", "assist": "assists", "poang": "points", "lag": "team", "laget": "team",
        "varfor": "why", "forlorar": "losing", "svit": "streak", "langsta": "longest", "flest": "most",
        "segrar": "wins", "byte": "trade", "kontrakt": "contract", "talang": "prospect", "ung": "young",
        "unga": "young", "laget": "my team", "ar sedan": "years ago", "sedan": "ago", "vann": "won", "pokalen": "stanley cup",
        "sasong": "season", "vilken": "which", "vi": "we", "skadad": "injury", "och": "and", "under": "under",
        "i": "on", "ligan": "league",
    },
    "fi": {
        "kuka": "who", "on": "is", "minun": "my", "meidan": "our", "paras": "best", "parhaat": "best",
        "huonoin": "worst", "pelaaja": "player", "pelaajat": "player", "puolustaja": "defenseman",
        "puolustajani": "my defenseman", "maalivahti": "goalie", "hyokkaaja": "forward", "sentteri": "center",
        "laitahyokkaaja": "wing", "maalit": "goals", "maali": "goal", "syotot": "assists", "pisteet": "points",
        "joukkue": "team", "joukkueeni": "my team", "miksi": "why", "haviamme": "losing", "putki": "streak",
        "voittoputki": "win streak", "pisin": "longest", "eniten": "most", "voitot": "wins", "vaihto": "trade",
        "sopimus": "contract", "lupaus": "prospect", "nuori": "young", "vuotta": "years", "sitten": "ago",
        "voitti": "won", "kausi": "season", "mika": "which", "me": "we", "loukkaantunut": "injury",
        "ja": "and", "alle": "under", "paras puolustajani": "my best defenseman", "liiga": "league", "liigan": "league",
    },
    "ru": {
        "кто": "who", "мой": "my", "моя": "my", "мои": "my", "моего": "my", "наш": "our", "наша": "our",
        "лучший": "best", "лучшая": "best", "лучшие": "best", "худший": "worst", "игрок": "player",
        "игроки": "player", "защитник": "defenseman", "защитники": "defenseman", "вратарь": "goalie",
        "нападающий": "forward", "центр": "center", "крайний": "wing", "голы": "goals", "гол": "goal",
        "голов": "goals", "передачи": "assists", "очки": "points", "очков": "points", "команда": "team",
        "команды": "team", "почему": "why", "проигрываем": "losing", "серия": "streak", "самая": "most",
        "длинная": "longest", "победы": "wins", "побед": "wins", "обмен": "trade", "контракт": "contract",
        "драфт": "draft", "проспект": "prospect", "молодой": "young", "лет": "years", "года": "years",
        "назад": "ago", "выиграл": "won", "кубок": "stanley cup", "сезон": "season", "какой": "which",
        "мы": "we", "травма": "injury", "сравни": "compare", "и": "and", "до": "under", "у": "", "в": "",
        "больше": "most", "всех": "", "забил": "scored", "лиги": "league", "лига": "league", "худший": "worst",
    },
}

LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "sv": "Swedish", "fi": "Finnish", "ru": "Russian"}

# Words that only one language uses, so a single hit is a strong language signal.
LANG_MARKERS: Dict[str, List[str]] = {
    "fr": ["qui", "mon", "meilleur", "pourquoi", "joueur", "equipe", "defenseur", "gardien", "quel", "notre"],
    "es": ["quien", "mejor", "jugador", "porque", "equipo", "defensor", "portero", "nuestro", "cual", "goles"],
    "de": ["wer", "ist", "bester", "warum", "spieler", "verteidiger", "torwart", "mannschaft", "welcher", "unser"],
    "sv": ["vem", "basta", "varfor", "spelare", "laget", "malvakt", "vilken", "backar", "poang"],
    "fi": ["kuka", "paras", "miksi", "pelaaja", "joukkue", "puolustaja", "maalivahti", "mika", "minun"],
}

# Common team nicknames the importer does not carry. Abbreviation -> extra names.
TEAM_NICKNAMES: Dict[str, List[str]] = {
    "TOR": ["leafs", "buds", "maple leafs"], "MTL": ["habs", "canadiens", "bleu blanc rouge", "les habitants"],
    "EDM": ["oilers", "oil"], "CGY": ["flames"], "VAN": ["canucks", "nucks"], "WPG": ["jets"],
    "OTT": ["sens", "senators"], "BOS": ["bruins", "bs"], "BUF": ["sabres"], "DET": ["wings", "red wings"],
    "TBL": ["bolts", "lightning", "tampa"], "FLA": ["panthers", "cats"], "NYR": ["rangers", "blueshirts"],
    "NYI": ["isles", "islanders"], "NJD": ["devils"], "PHI": ["flyers"], "PIT": ["pens", "penguins"],
    "WSH": ["caps", "capitals"], "CAR": ["canes", "hurricanes"], "CBJ": ["jackets", "blue jackets", "cbj"],
    "CHI": ["hawks", "blackhawks"], "STL": ["blues"], "NSH": ["preds", "predators"], "DAL": ["stars"],
    "COL": ["avs", "avalanche"], "MIN": ["wild"], "UTA": ["utah", "mammoth", "hockey club"],
    "VGK": ["knights", "golden knights", "vegas"], "LAK": ["kings", "la"], "ANA": ["ducks"],
    "SJS": ["sharks"], "SEA": ["kraken"], "ARI": ["coyotes", "yotes"],
}


# Everyday English words. They keep plain words from being "corrected" into
# player surnames ("love" -> Love, "money" -> Mooney) and count as understood.
COMMON_WORDS = set("""
a about above after again against all almost also always am an and any anyone anything are around as ask at away
back bad be because been before being below best better between big both but by call came can cant could day did
do does doing done down during each early either else end enough even ever every everyone everything far few find
first for from full get give go goes going gone good got great had has have having he her here him his how i if in
into is it its just keep kind know last least left less let like little long look lot love made make making many
may me might mine more most much must my need never new next no not nothing now of off often old on once one only
or other our out over own part people play played player please put rather really right same say see seem she
should show since so some something soon still such sure take tell than thank thanks that the their them then there
these they thing things think this those though through time to today together too took top toward try turn under
until up upon us use used very want was way we well went were what when where which while who whole whom whose why
will with without won work would yeah yes yet you your hockey money team game games job safe fired weather joke love
win wins winning cup record predict prediction attendance commissioner owner tank tanking average middle pack every
rank ranking ranked mid closest ready younger older cheaper cheapest expensive priciest exactly year years old age
legend legends hall fame famers retired numbers instructions ignore previous everyone table drop hack cheat stats
three four five six seven eight nine ten currently current provide provides providing project projects
projected relative relatively compared comparable comparables expected expectations since despite receiving
benefit benefiting benefits suggest suggests upcoming increase decrease likely unlikely realistically realistic
significantly entire whole across within against among between during over under above below
largest biggest longest shortest highest lowest greatest fastest slowest strongest weakest newest
anymore anyway actually really scores score scored scoring cheaply cheaper afford affordable brother brothers sister family stalling stall stalled stuck growing grow grew develop developing
asset assets spend spending become becoming getting get gets slower slow faster fast quicker hurt hurts hurting asking
ask asked earlier before later steal steals stealing bust busts drafted outside inside round rounds overpay overpays
overpaying overpaid desperate value values valued performer performers performance talks talk talking nobody
somebody anybody division conference ever had has have playoff playoffs miss missing missed every each year years
improve improving improved losing lose lost loss without spending tougher tough toughest worth trading trade traded
down up move moved moving switch side right left offense offence offensive defense defence defensive goaltending
goaltender goalie goalies untouchable untouchables except everyone everybody blockbuster realistic least most hurts
attendance falling fall dropping drop drops morale unhappy happy mad upset angry wants want out leave leaving stay
home stay-at-home shutdown mover puck pucks beauty mail carries carry carrying steals drives drive driving jam grit
gritty locker room guy guys fit fits fitting roster lineup pair pairing partner current compare compared versus
contract contracts cap space money paid pay salary deal deals available still free agent agents signed unsigned
rights draft drafts pick picks class classes prospect prospects junior juniors floor ceiling upside fail failure
likely closest ready later rebuild rebuilding contend contending competitive veterans veteran young younger older
old oldest youngest age profitable profit immediately quickly now soon franchise history biggest goal goals
upset upsets season seasons streak streaks point points captain captains played games game record records
almost made happened last show again those these them they their there that this which what who whom whose why
how when where would could should might must can cant will wont did does do done doing better best worse worst
good great bad terrible awful decent average elite top bottom first second third fourth line lines skater skaters
""".split())
