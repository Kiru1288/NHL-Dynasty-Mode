"""Extra name pools for generated players.

The base pools in name_generator held ~30 first / ~30 last names per country, so a
league of 6,000+ generated players repeated the same names constantly. They also
mixed in famous NHL surnames ("Nathan McDavid", "Kaeden McDavid"). These lists widen
the main hockey nations several-fold and drop distinctive star surnames.
"""

from typing import Dict, List

# Distinctive NHL / hockey-legend surnames that read as a joke on a generated player.
SURNAME_BLACKLIST = {
    "Crosby", "McDavid", "Stamkos", "Marchand", "Barzal", "Reinhart", "Sanderson", "Cirelli",
    "Suzuki", "Byfield", "Nugent-Hopkins", "MacKinnon", "Toews", "Scheifele", "Duclair",
    "Comtois", "Veleno", "Perfetti", "Giroux", "Bergeron", "Fleury", "Point",
    "Lundqvist", "Forsberg", "Hedlund",
    "Ovechkin", "Malkin", "Kovalchuk", "Zadorov",
    "Jagr", "Hasek", "Pavelec", "Zacha", "Hronek", "Kampf", "Palat", "Krejci",
    "Chara", "Hossa", "Gaborik", "Demitra", "Slafkovsky", "Tatár", "Fehervary", "Cernak", "Sekera",
    "Josi", "Hischier", "Niederreiter", "Ambühl",
    "Girgensons", "Merzlikins", "Kivlenieks", "Daugavins", "Znaroks", "Skrastins", "Bukarts", "Indrasis",
    "Rantanen", "Laine", "Hakala",
}

EXTRA_NAMES: Dict[str, Dict[str, List[str]]] = {
    "Canada": {
        "first": [
            "Adam","Aaron","Austin","Blake","Brady","Brett","Brody","Calvin","Cody","Curtis",
            "Dalton","Damien","Dante","Derek","Dominic","Drew","Easton","Eric","Evan","Finn",
            "Gage","Graham","Grant","Harrison","Hayden","Hunter","Ian","Jace","Jared","Jesse",
            "Joel","Jonah","Josh","Julien","Justin","Kade","Keegan","Kieran","Kyle","Lane",
            "Liam","Luc","Marc","Marcus","Mathieu","Micah","Mitchell","Nash","Nico","Parker",
            "Patrick","Philippe","Quinn","Reese","Rhys","Rowan","Sebastian","Shane","Simon","Spencer",
            "Steven","Teddy","Tristan","Trevor","Vincent","Zach","Zane","Alexis","Olivier","Jérémy",
            "Louis","Mathis","Hugo","Raphaël","William","Xavier","Cédric","Dawson","Brendan","Colin",
        ],
        "last": [
            "Abbott","Archibald","Arsenault","Aucoin","Barrett","Beauchamp","Bélanger","Bennett","Bergevin","Bishop",
            "Blackwood","Boivin","Boucher","Bourque","Boyle","Bradley","Brennan","Brodeur","Burke","Cameron",
            "Caron","Carrier","Chiasson","Chisholm","Cloutier","Collins","Cormier","Cousins","Cummings","Daigle",
            "Desjardins","Dion","Doherty","Donnelly","Doucet","Dufresne","Dumont","Dunn","Elliott","Fairbairn",
            "Ferguson","Fitzgerald","Forbes","Fournier","Fraser","Gallant","Gaudreau","Gill","Goulet","Graham",
            "Grant","Guérin","Hall","Harper","Hebert","Henderson","Hogan","Irving","Jacobs","Kennedy",
            "Kerr","Labelle","Lachance","Lafleur","Lamb","Lambert","Landry","Langlois","Lapointe","Laroche",
            "Leblanc","Leclerc","Lemieux","Lessard","Lévesque","MacAulay","MacInnis","MacIsaac","MacLean","MacNeil",
            "MacPherson","Maltais","Marsh","McCarthy","McGregor","McIntyre","McKay","McLean","McNeil","Mercier",
            "Michaud","Moore","Morin","Morrison","Munro","Murray","Nadeau","Nicholson","Noël","Ouellet",
            "Paquette","Paradis","Parent","Patterson","Pearson","Perreault","Picard","Plante","Poirier","Poulin",
            "Prentice","Proulx","Quinn","Rankin","Richard","Robichaud","Robertson","Rousseau","Saunders","Savard",
            "Sinclair","Simard","Stewart","Sullivan","Sutherland","Thibault","Thibodeau","Turcotte","Turner","Vachon",
            "Vaillancourt","Vézina","Wallace","Watson","Whelan","Whitfield","Woods","Yeo","Babcock","Bastien",
        ],
        "towns": ["Sudbury, ON","Barrie, ON","Oshawa, ON","Kingston, ON","Peterborough, ON","Sault Ste. Marie, ON","Thunder Bay, ON","Kelowna, BC","Kamloops, BC","Prince George, BC","Red Deer, AB","Lethbridge, AB","Medicine Hat, AB","Brandon, MB","Moncton, NB","Saint John, NB","Charlottetown, PE","Sherbrooke, QC","Trois-Rivières, QC","Rimouski, QC"],
    },
    "USA": {
        "first": [
            "Aidan","Blake","Bobby","Brady","Brendan","Brock","Caleb","Cameron","Casey","Chase",
            "Christian","Clayton","Cooper","Corey","Dalton","Danny","Derek","Drew","Dustin","Eli",
            "Eric","Garrett","Grant","Hank","Hunter","Isaac","Jackson","Jason","Jesse","Jordan",
            "Kyle","Liam","Mason","Max","Mitchell","Nate","Owen","Parker","Patrick","Quinn",
            "Riley","Sam","Sean","Seth","Shane","Spencer","Thomas","Trevor","Troy","Will",
        ],
        "last": [
            "Adams","Allen","Bailey","Baker","Barnes","Bell","Bennett","Brooks","Bryant","Butler",
            "Campbell","Carlson","Carter","Coleman","Collins","Cook","Cooper","Cox","Crawford","Cunningham",
            "Doyle","Duffy","Dwyer","Edwards","Evans","Fitzpatrick","Flanagan","Foley","Foster","Gallagher",
            "Graham","Gray","Griffin","Hayes","Healy","Henderson","Hoffman","Howard","Hughes","Hunt",
            "Jensen","Kelly","Kennedy","Lynch","Mahoney","McCarthy","McGuire","Meyer","Mitchell","Morgan",
            "Murphy","Nelson","O'Brien","O'Connor","Olson","Parker","Peterson","Phillips","Powell","Quinn",
            "Reed","Reilly","Richardson","Rogers","Ross","Russell","Ryan","Schultz","Shea","Sullivan",
            "Sweeney","Turner","Walsh","Ward","Watson","Wheeler","Wood","Wright","Young","Zimmerman",
        ],
        "towns": ["Grand Rapids, MI","Green Bay, WI","Madison, WI","Duluth, MN","St. Paul, MN","Rochester, NY","Syracuse, NY","Hartford, CT","Providence, RI","Portland, ME","Anchorage, AK","Fargo, ND","Columbus, OH","Philadelphia, PA","Dallas, TX","Phoenix, AZ","Nashville, TN","Raleigh, NC","Tampa, FL","St. Louis, MO"],
    },
    "Sweden": {
        "first": [
            "Albin","Alexander","Alfons","Anton","Arvid","Axel","Calle","Daniel","David","Elias",
            "Emil","Erik","Filip","Fredrik","Gustav","Hampus","Hugo","Isak","Jacob","Jesper",
            "Joel","Johan","Jonas","Kalle","Leo","Linus","Loke","Lucas","Ludvig","Marcus",
            "Mattias","Max","Melker","Nils","Noah","Oliver","Oscar","Otto","Pontus","Rasmus",
            "Robin","Sebastian","Simon","Theo","Tim","Viktor","Vilgot","Wilhelm","Adrian","Elton",
        ],
        "last": [
            "Ahlberg","Almqvist","Åkesson","Axelsson","Backman","Bengtsson","Berg","Blomqvist","Bodin","Borg",
            "Dahlberg","Danielsson","Ek","Engberg","Falk","Fransson","Fredriksson","Grahn","Granlund","Hagberg",
            "Hallberg","Hedberg","Hellström","Henriksson","Holmberg","Holmgren","Isaksson","Jakobsson","Jonsson","Kjellberg",
            "Lindahl","Lindberg","Lindholm","Ljung","Lund","Lundberg","Lundin","Magnusson","Malm","Mattsson",
            "Molin","Nordin","Nordström","Norberg","Nyberg","Öberg","Palm","Rosén","Samuelsson","Sjöström",
            "Sköld","Sundberg","Sundin","Söderberg","Ström","Thorén","Wallin","Wennberg","Westerlund","Wikström",
        ],
        "towns": ["Stockholm","Gothenburg","Malmö","Västerås","Örebro","Linköping","Luleå","Skellefteå","Umeå","Växjö","Jönköping","Karlstad","Gävle","Södertälje","Örnsköldsvik"],
    },
    "Finland": {
        "first": [
            "Aapo","Aarne","Aatu","Aleksi","Anton","Arttu","Eelis","Eemeli","Eero","Elias",
            "Henri","Ilari","Jaakko","Jere","Jesse","Joel","Joni","Juho","Jussi","Kalle",
            "Kasper","Lauri","Leevi","Markus","Matias","Mikko","Miro","Niko","Oliver","Onni",
            "Otto","Patrik","Roope","Sami","Santeri","Severi","Topi","Tuomas","Urho","Veeti",
        ],
        "last": [
            "Ahonen","Anttila","Halonen","Harju","Heikkilä","Hiltunen","Hirvonen","Huttunen","Ikonen","Jokela",
            "Kangas","Kauppinen","Kemppainen","Kettunen","Kivelä","Kokko","Kolehmainen","Korpela","Kurki","Laakso",
            "Lahtinen","Laitinen","Lampinen","Lappalainen","Leinonen","Leppänen","Lindroos","Manninen","Miettinen","Mustonen",
            "Niemi","Nurminen","Ojala","Partanen","Pesonen","Pitkänen","Pulkkinen","Rautio","Riihimäki","Ruotsalainen",
            "Salonen","Seppälä","Sirola","Suominen","Tikkanen","Turunen","Väisänen","Valtonen","Vesterinen","Viljanen",
        ],
        "towns": ["Helsinki","Tampere","Turku","Oulu","Espoo","Jyväskylä","Lahti","Kuopio","Pori","Hämeenlinna","Rauma","Lappeenranta","Vaasa","Joensuu"],
    },
    "Russia": {
        "first": [
            "Aleksei","Anatoli","Andrei","Anton","Artyom","Bogdan","Daniil","Denis","Dmitri","Egor",
            "Fyodor","Gleb","Grigori","Ilya","Ivan","Kirill","Konstantin","Leonid","Makar","Matvei",
            "Maxim","Mikhail","Nikita","Nikolai","Oleg","Pavel","Roman","Ruslan","Semyon","Sergei",
            "Stepan","Timofei","Vadim","Valeri","Vasili","Viktor","Vladislav","Yaroslav","Yegor","Yuri",
        ],
        "last": [
            "Abramov","Afanasiev","Alekseev","Antonov","Baranov","Belov","Bobrov","Borisov","Bykov","Chernov",
            "Danilov","Davydov","Denisov","Dmitriev","Efimov","Egorov","Ershov","Filippov","Frolov","Gavrilov",
            "Gerasimov","Golubev","Gordeev","Grigoriev","Ilyin","Isaev","Kalinin","Kazakov","Klimov","Komarov",
            "Kondratiev","Korolev","Krylov","Kudryavtsev","Larionov","Lazarev","Maksimov","Medvedev","Melnikov","Mironov",
            "Nazarov","Nikitin","Osipov","Panov","Polyakov","Romanov","Rybakov","Shcherbakov","Sidorov","Stepanov",
            "Titov","Tikhonov","Trofimov","Vinogradov","Vlasov","Voronin","Yakovlev","Zakharov","Zhukov","Zuev",
        ],
        "towns": ["Moscow","St. Petersburg","Yaroslavl","Kazan","Ufa","Omsk","Magnitogorsk","Chelyabinsk","Nizhny Novgorod","Yekaterinburg","Novosibirsk","Cherepovets","Tolyatti","Khabarovsk"],
    },
    "Czechia": {
        "first": [
            "Adam","Ales","Daniel","David","Dominik","Filip","Frantisek","Jakub","Jan","Jaroslav",
            "Jiri","Josef","Kryštof","Lukas","Marek","Martin","Matej","Michal","Ondrej","Patrik",
            "Pavel","Petr","Radek","Roman","Stepan","Tomas","Vaclav","Vit","Vojtech","Zdenek",
        ],
        "last": [
            "Bartak","Beran","Bures","Cech","Chalupa","Dolezal","Dusek","Fiser","Hajek","Havel",
            "Holub","Hruby","Janda","Kadlec","Kopecky","Kral","Krejcik","Kriz","Malek","Marek",
            "Matousek","Moravec","Musil","Nemec","Polak","Pospisil","Riha","Sedlacek","Sedlak","Soukup",
            "Stastny","Sykora","Tichy","Urban","Vacek","Vanek","Vlcek","Vondracek","Zeman","Zima",
        ],
        "towns": ["Prague","Brno","Ostrava","Pilsen","Liberec","Olomouc","Pardubice","Hradec Kralove","Zlin","Kladno","Trinec","Vitkovice","Litvinov","Jihlava"],
    },
    "Slovakia": {
        "first": [
            "Adam","Adrian","Andrej","Boris","Daniel","Dominik","Filip","Jakub","Jan","Jozef",
            "Juraj","Kristian","Libor","Lukas","Marek","Martin","Matej","Michal","Milan","Miroslav",
            "Patrik","Pavol","Peter","Rastislav","Richard","Samuel","Simon","Stefan","Tomas","Viliam",
        ],
        "last": [
            "Bednar","Benko","Blasko","Bodnar","Cervenka","Fabian","Galik","Hanzel","Hudak","Jancek",
            "Kmet","Kollar","Kopecky","Kostka","Lacko","Lukac","Marko","Matus","Meszaros","Mihalik",
            "Novotny","Oravec","Pavlik","Polak","Sedlak","Slovak","Stefanik","Strba","Svec","Vavrinec",
        ],
        "towns": ["Bratislava","Kosice","Zilina","Nitra","Banska Bystrica","Trencin","Poprad","Presov","Zvolen","Michalovce","Liptovsky Mikulas","Martin"],
    },
    "Germany": {
        "first": [
            "Alexander","Ben","Constantin","Dominik","Elias","Felix","Finn","Florian","Jan","Jannik",
            "Jonas","Julian","Kai","Korbinian","Leon","Lennart","Luca","Lukas","Marco","Max",
            "Moritz","Nico","Niklas","Noah","Paul","Philipp","Sebastian","Simon","Tim","Tobias",
        ],
        "last": [
            "Albrecht","Arnold","Bauer","Beck","Brandt","Busch","Dietrich","Ernst","Frank","Friedrich",
            "Graf","Günther","Haas","Hahn","Heinrich","Herrmann","Horn","Jung","Keller","König",
            "Kraus","Kühn","Lang","Lorenz","Ludwig","Martin","Mayer","Möller","Otto","Pohl",
            "Roth","Sauer","Schmitt","Schubert","Schuster","Stein","Vogt","Walter","Werner","Winter",
        ],
        "towns": ["Munich","Berlin","Cologne","Mannheim","Düsseldorf","Hamburg","Nuremberg","Augsburg","Ingolstadt","Wolfsburg","Krefeld","Iserlohn","Straubing","Bremerhaven"],
    },
    "Switzerland": {
        "first": [
            "Andrin","Dario","Denis","Fabian","Gaëtan","Janis","Jonas","Kevin","Lars","Leandro",
            "Lian","Luca","Marco","Mattia","Nico","Noah","Pius","Reto","Sandro","Simon",
            "Sven","Thierry","Timo","Valentin","Yannick",
        ],
        "last": [
            "Ammann","Bianchi","Bosshard","Brügger","Egli","Fehr","Frey","Hofer","Hofmann","Kälin",
            "Kunz","Lüthi","Marti","Moser","Müller","Rüegg","Schaller","Steiner","Vogel","Widmer",
            "Wyss","Zaugg","Zimmermann","Zwahlen","Lehmann",
        ],
        "towns": ["Zurich","Bern","Lugano","Davos","Geneva","Lausanne","Fribourg","Biel","Zug","Langnau","Ambri","Kloten"],
    },
}


def merge_name_pools(base: Dict[str, Dict[str, List[str]]]) -> None:
    """Merge EXTRA_NAMES into ``base`` in place, dedupe, and drop blacklisted surnames."""
    for country, extra in EXTRA_NAMES.items():
        pool = base.setdefault(country, {"first": [], "last": [], "towns": []})
        for key in ("first", "last", "towns"):
            seen = set(pool.get(key) or [])
            merged = list(pool.get(key) or [])
            for name in extra.get(key) or []:
                if name not in seen:
                    seen.add(name)
                    merged.append(name)
            pool[key] = merged
    for pool in base.values():
        lasts = [n for n in (pool.get("last") or []) if n not in SURNAME_BLACKLIST]
        if lasts:
            pool["last"] = lasts
