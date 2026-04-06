"""
Human name + nationality + hometown generator.

Requirements implemented:
- global hockey + non-traditional markets
- culturally sensible name structures (best-effort)
- nickname probability
- optional pronunciation key

All functions accept a `random.Random` instance for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _safe_choice(rng, items: List[str]) -> str:
    if not items:
        return "Unknown"
    return str(rng.choice(items))


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass(frozen=True)
class HumanIdentity:
    full_name: str
    nationality: str
    hometown: str
    nickname: Optional[str] = None
    pronunciation: Optional[str] = None


# ---------------------------------------------------------------------------
# Country pools (30+ first + 30+ last each)
# ---------------------------------------------------------------------------

NAME_POOLS: Dict[str, Dict[str, List[str]]] = {
    "Canada": {
        "first": [
            "Liam","Noah","Oliver","Ethan","Jacob","Lucas","Logan","Jackson","Aiden","Benjamin",
            "William","Mason","Carter","Wyatt","Owen","Hudson","Leo","Caleb","Dylan","Nathan",
            "Ryan","Cole","Connor","Tyler","Brayden","Matthew","Zachary","Jordan","Alex","Jake",
        ],
        "last": [
            "Smith","Johnson","Brown","Wilson","Campbell","MacDonald","Taylor","Anderson","Thompson","Clark",
            "Martin","Lee","Young","Walker","King","Wright","Scott","Green","Baker","Adams",
            "Hughes","Miller","Crosby","McDavid","Dubois","Roy","Giroux","Bouchard","Fleury","McKenzie",
        ],
        "towns": ["Toronto, ON","Montreal, QC","Vancouver, BC","Calgary, AB","Edmonton, AB","Ottawa, ON","Winnipeg, MB","Halifax, NS","Quebec City, QC","Saskatoon, SK"],
    },
    "USA": {
        "first": [
            "James","Michael","David","Joseph","Daniel","Matthew","Andrew","Joshua","Christopher","Anthony",
            "John","Ryan","Nicholas","Brandon","Tyler","Zachary","Evan","Logan","Benjamin","Carter",
            "Dylan","Connor","Jack","Luke","Noah","Alex","Jake","Cole","Austin","Kevin",
        ],
        "last": [
            "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
            "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
            "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
        ],
        "towns": ["Minneapolis, MN","Boston, MA","Chicago, IL","Detroit, MI","Buffalo, NY","Denver, CO","Pittsburgh, PA","New York, NY","Los Angeles, CA","Seattle, WA"],
    },
    "Sweden": {
        "first": [
            "Elias","William","Oscar","Alexander","Lucas","Hugo","Oliver","Liam","Noah","Emil",
            "Filip","Viktor","Joel","Anton","Isak","Simon","Gustav","Linus","Albin","Marcus",
            "Sebastian","Johan","Erik","Karl","Adam","Nils","Jesper","Henrik","Fredrik","Axel",
        ],
        "last": [
            "Johansson","Andersson","Karlsson","Nilsson","Eriksson","Larsson","Olsson","Persson","Svensson","Gustafsson",
            "Pettersson","Bergström","Lindström","Lindgren","Berglund","Håkansson","Sandberg","Holm","Dahl","Björk",
            "Forsberg","Sjöberg","Ekström","Lundqvist","Lindqvist","Engström","Bergman","Nyström","Strömberg","Hedlund",
        ],
        "towns": ["Stockholm","Gothenburg","Malmö","Uppsala","Linköping","Västerås","Örebro","Luleå","Karlstad","Jönköping"],
    },
    "Finland": {
        "first": [
            "Mikko","Aleksi","Jussi","Jani","Sami","Jere","Rasmus","Eetu","Olli","Juho",
            "Antti","Teemu","Ville","Markus","Tuomas","Elias","Joel","Henri","Kasper","Niko",
            "Jukka","Onni","Arttu","Lauri","Valtteri","Matias","Joonas","Toni","Petteri","Ilkka",
        ],
        "last": [
            "Korhonen","Virtanen","Mäkinen","Nieminen","Mäkelä","Hämäläinen","Laine","Heikkinen","Koskinen","Järvinen",
            "Lehtinen","Lehtonen","Saarinen","Salminen","Heinonen","Kinnunen","Rantanen","Toivonen","Tuominen","Salo",
            "Karjalainen","Aaltonen","Hakala","Kallio","Räsänen","Sundqvist","Peltola","Nurmi","Kujala","Keto",
        ],
        "towns": ["Helsinki","Tampere","Turku","Oulu","Jyväskylä","Kuopio","Lahti","Pori","Lappeenranta","Vaasa"],
    },
    "Russia": {
        "first": [
            "Alexei","Ivan","Dmitri","Sergei","Nikita","Andrei","Mikhail","Pavel","Vladimir","Kirill",
            "Artem","Ilya","Yegor","Denis","Roman","Viktor","Oleg","Maxim","Yuri","Konstantin",
            "Anton","Alexander","Semyon","Grigori","Timur","Stanislav","Vadim","Gennady","Fedor","Matvei",
        ],
        "last": [
            "Ivanov","Smirnov","Kuznetsov","Popov","Sokolov","Lebedev","Kozlov","Novikov","Morozov","Petrov",
            "Volkov","Solovyov","Vasiliev","Zaitsev","Pavlov","Semenov","Bogdanov","Gusev","Orlov","Makarov",
            "Fedorov","Karpov","Nikolaev","Kiselev","Tarasov","Belyaev","Sorokin","Kulikov","Zadorov","Kovalchuk",
        ],
        "towns": ["Moscow","Saint Petersburg","Yekaterinburg","Kazan","Novosibirsk","Chelyabinsk","Omsk","Ufa","Nizhny Novgorod","Samara"],
    },
    "Czechia": {
        "first": [
            "Jan","Petr","Martin","Jakub","Tomas","David","Michal","Jiri","Adam","Filip",
            "Lukas","Ondrej","Matej","Vojtech","Roman","Karel","Dominik","Daniel","Josef","Radek",
            "Patrik","Marek","Viktor","Stanislav","Jaroslav","Vaclav","Aleš","Zdenek","Bohuslav","Antonin",
        ],
        "last": [
            "Novak","Svoboda","Novotny","Dvorak","Cerny","Prochazka","Kucera","Vesely","Horak","Nemecek",
            "Pokorny","Krejci","Fiala","Simek","Kolar","Ruzicka","Jelinek","Blaha","Bartos","Benes",
            "Machacek","Kovar","Hasek","Pavelec","Stransky","Zacha","Hronek","Kampf","Jagr","Palat",
        ],
        "towns": ["Prague","Brno","Ostrava","Plzeň","Liberec","Olomouc","Pardubice","České Budějovice","Hradec Králové","Zlín"],
    },
    "Slovakia": {
        "first": [
            "Martin","Peter","Marek","Tomas","Juraj","Michal","Filip","Jakub","Adam","Robert",
            "Daniel","Matej","Andrej","Jozef","Lukas","Dominik","Viktor","Ivan","Richard","Stanislav",
            "Stefan","Patrik","Radoslav","Tibor","Boris","Erik","Milan","Samuel","Jan","Karol",
        ],
        "last": [
            "Novak","Horvath","Varga","Kovac","Toth","Nagy","Balaz","Krajci","Duda","Kral",
            "Simek","Urban","Mikula","Sykora","Cernak","Chara","Hossa","Gaborik","Tatár","Fehervary",
            "Hrivik","Laco","Budaj","Demitra","Sekera","Hudacek","Lunter","Jurco","Panik","Slafkovsky",
        ],
        "towns": ["Bratislava","Košice","Prešov","Žilina","Nitra","Trnava","Banská Bystrica","Trenčín","Martin","Poprad"],
    },
    "Germany": {
        "first": [
            "Leon","Lukas","Finn","Jonas","Paul","Noah","Elias","Felix","Maximilian","Tim",
            "Moritz","Ben","Niklas","Jan","David","Fabian","Philipp","Tobias","Kevin","Marcel",
            "Daniel","Sebastian","Florian","Simon","Alexander","Julian","Jannik","Nico","Oliver","Markus",
        ],
        "last": [
            "Müller","Schmidt","Schneider","Fischer","Weber","Meyer","Wagner","Becker","Hoffmann","Schäfer",
            "Koch","Bauer","Richter","Klein","Wolf","Schröder","Neumann","Schwarz","Zimmermann","Braun",
            "Hartmann","Krüger","Maier","Lehmann","Kaiser","Franke","Vogel","Peters","Schulz","Seidel",
        ],
        "towns": ["Berlin","Munich","Hamburg","Cologne","Frankfurt","Düsseldorf","Mannheim","Nuremberg","Augsburg","Stuttgart"],
    },
    "Switzerland": {
        "first": [
            "Luca","Noah","Leon","Jan","Nico","Samuel","Simon","David","Jonas","Elias",
            "Matteo","Fabio","Marco","Kevin","Adrian","Benjamin","Dominic","Pascal","Tim","Oliver",
            "Daniel","Lars","Sebastian","Philipp","Michael","Roman","Tobias","Stefan","Damian","Silvan",
        ],
        "last": [
            "Müller","Meier","Schmid","Keller","Weber","Fischer","Brunner","Suter","Baumann","Graf",
            "Schneider","Huber","Gerber","Frei","Bachmann","Roth","Gasser","Bieri","Wicki","Kaufmann",
            "Zürcher","Frick","Bühler","Streit","Josi","Hischier","Niederreiter","Fiala","Ambühl","Geering",
        ],
        "towns": ["Zürich","Geneva","Basel","Bern","Lausanne","Lucerne","St. Gallen","Lugano","Biel/Bienne","Winterthur"],
    },
    "France": {
        "first": [
            "Louis","Jules","Gabriel","Hugo","Lucas","Arthur","Nathan","Ethan","Léo","Tom",
            "Paul","Théo","Enzo","Noah","Maxime","Antoine","Mathis","Clément","Nicolas","Alexandre",
            "Baptiste","Romain","Adrien","Kevin","Simon","Victor","Damien","Florian","Yoan","Pierre",
        ],
        "last": [
            "Martin","Bernard","Thomas","Petit","Robert","Richard","Durand","Dubois","Moreau","Laurent",
            "Simon","Michel","Lefebvre","Leroy","Roux","David","Bertrand","Morel","Fournier","Girard",
            "Bonnet","Dupont","Lambert","Fontaine","Rousseau","Vincent","Muller","Lefevre","Garcia","Chevalier",
        ],
        "towns": ["Paris","Lyon","Marseille","Toulouse","Nice","Grenoble","Strasbourg","Bordeaux","Lille","Rouen"],
    },
    "UK": {
        "first": [
            "Oliver","George","Harry","Jack","Jacob","Noah","Charlie","Thomas","Oscar","William",
            "James","Henry","Leo","Alfie","Joshua","Freddie","Ethan","Alexander","Samuel","Daniel",
            "Joseph","Matthew","Adam","Ryan","Luke","Callum","Ben","Nathan","Connor","Edward",
        ],
        "last": [
            "Smith","Jones","Taylor","Brown","Williams","Wilson","Johnson","Davies","Robinson","Wright",
            "Thompson","Evans","Walker","White","Hall","Green","Lewis","Clarke","Wood","Harris",
            "Baker","Cooper","King","Ward","Turner","Parker","Price","Morgan","Reid","Scott",
        ],
        "towns": ["London","Manchester","Birmingham","Leeds","Glasgow","Edinburgh","Sheffield","Bristol","Cardiff","Nottingham"],
    },
    "Norway": {
        "first": [
            "Noah","Oliver","Emil","Lucas","Jakob","Oskar","Aksel","William","Elias","Magnus",
            "Theodor","Matias","Sander","Henrik","Jonas","Mathias","Tobias","Kristian","Andreas","Martin",
            "Filip","Marius","Liam","Isak","Benjamin","Sebastian","Johan","Erik","Håkon","Nils",
        ],
        "last": [
            "Hansen","Johansen","Olsen","Larsen","Andersen","Pedersen","Nilsen","Kristiansen","Jensen","Karlsen",
            "Johnsen","Pettersen","Eriksen","Berg","Haugen","Hagen","Moen","Bakken","Aas","Sund",
            "Solberg","Eide","Lie","Foss","Sæther","Mikkelsen","Nygaard","Dahl","Bjerke","Strand",
        ],
        "towns": ["Oslo","Bergen","Trondheim","Stavanger","Drammen","Fredrikstad","Kristiansand","Tromsø","Skien","Ålesund"],
    },
    "Denmark": {
        "first": [
            "William","Noah","Emil","Oliver","Lucas","Malthe","Oscar","Frederik","Victor","Magnus",
            "Mathias","Alexander","Sebastian","Mikkel","Jens","Andreas","Jonas","Tobias","Kristian","Nikolaj",
            "Elias","Liam","Simon","Benjamin","Anton","Casper","Rasmus","David","Philip","Oliver",
        ],
        "last": [
            "Jensen","Nielsen","Hansen","Pedersen","Andersen","Christensen","Larsen","Sørensen","Rasmussen","Jørgensen",
            "Petersen","Madsen","Kristensen","Olsen","Thomsen","Poulsen","Knudsen","Mortensen","Jakobsen","Mikkelsen",
            "Frederiksen","Eriksen","Schmidt","Højgaard","Bach","Brandt","Holm","Lund","Villadsen","Skov",
        ],
        "towns": ["Copenhagen","Aarhus","Odense","Aalborg","Esbjerg","Randers","Kolding","Horsens","Vejle","Roskilde"],
    },
    "Latvia": {
        "first": [
            "Janis","Martins","Kristaps","Rihards","Edgars","Andris","Arturs","Roberts","Niks","Toms",
            "Davis","Emils","Aleksandrs","Miks","Reinis","Oskars","Gatis","Raitis","Ernests","Harijs",
            "Vilnis","Lauris","Matiss","Sandis","Renars","Gustavs","Rudolfs","Karlis","Uvis","Aigars",
        ],
        "last": [
            "Berzins","Kalnins","Ozols","Jansons","Liepa","Ziedins","Balodis","Vilks","Krastins","Eglitis",
            "Krumins","Kuzmins","Petrovs","Sokolovs","Cibuls","Girgensons","Biedrins","Daugavins","Merzlikins","Kivlenieks",
            "Indrasis","Bukarts","Skrastins","Karsums","Kuleshovs","Pavlovs","Ravins","Silins","Znaroks","Nieder",
        ],
        "towns": ["Riga","Daugavpils","Liepāja","Jelgava","Jūrmala","Ventspils","Rēzekne","Valmiera","Jēkabpils","Ogre"],
    },
    "Belarus": {
        "first": [
            "Ivan","Dmitry","Sergey","Andrei","Nikita","Ilya","Alexei","Pavel","Artem","Maxim",
            "Kirill","Roman","Vladislav","Denis","Yegor","Oleg","Mikhail","Konstantin","Viktor","Yuri",
            "Anton","Alexander","Stanislav","Timofey","Gleb","Fedor","Stepan","Semyon","Valery","Igor",
        ],
        "last": [
            "Ivanov","Kozlov","Novikov","Smirnov","Kuznetsov","Morozov","Sokolov","Petrov","Volkov","Orlov",
            "Solovyov","Pavlov","Belyaev","Kiselev","Tarasov","Karpov","Gusev","Sorokin","Zaitsev","Makarov",
            "Sharanovich","Protas","Kolya","Kostitsyn","Saley","Grabovski","Kalyuzhny","Mezin","Yermakov","Ugarov",
        ],
        "towns": ["Minsk","Gomel","Mogilev","Vitebsk","Grodno","Brest","Bobruisk","Baranovichi","Borisov","Pinsk"],
    },
    "Ukraine": {
        "first": [
            "Oleksandr","Andrii","Dmytro","Ivan","Maksym","Artem","Bohdan","Mykola","Vladyslav","Serhii",
            "Yurii","Roman","Taras","Denys","Ihor","Vitalii","Pavlo","Oleksii","Kyrylo","Sviatoslav",
            "Oleh","Stepan","Vasyl","Yevhen","Nazar","Ruslan","Volodymyr","Hryhorii","Marko","Lev",
        ],
        "last": [
            "Shevchenko","Kovalenko","Bondarenko","Tkachenko","Kovalchuk","Boyko","Kravchenko","Oliynyk","Melnyk","Moroz",
            "Pavlenko","Savchenko","Petrenko","Lysenko","Ivanenko","Danylenko","Polishchuk","Sydorenko","Hrytsenko","Marchenko",
            "Yaremchuk","Zakharchenko","Rudenko","Klymenko","Maksymenko","Fedorenko","Havrylenko","Kucherenko","Semenko","Korniyenko",
        ],
        "towns": ["Kyiv","Kharkiv","Lviv","Dnipro","Odesa","Zaporizhzhia","Kryvyi Rih","Mykolaiv","Vinnytsia","Chernihiv"],
    },
    "Poland": {
        "first": [
            "Jakub","Jan","Szymon","Kacper","Mateusz","Adam","Mikolaj","Pawel","Piotr","Michal",
            "Tomasz","Karol","Filip","Marcin","Wojciech","Bartosz","Damian","Igor","Rafal","Oskar",
            "Kamil","Dawid","Maciej","Lukasz","Patryk","Sebastian","Konrad","Antoni","Dominik","Hubert",
        ],
        "last": [
            "Nowak","Kowalski","Wisniewski","Wojcik","Kowalczyk","Kaminski","Lewandowski","Zielinski","Szymanski","Wozniak",
            "Dabrowski","Kozlowski","Jankowski","Mazur","Krawczyk","Piotrowski","Grabowski","Pawlak","Michalski","Nowicki",
            "Adamczyk","Dudek","Zajac","Wieczorek","Walczak","Baran","Stepien","Kubiak","Borkowski","Chmielewski",
        ],
        "towns": ["Warsaw","Kraków","Gdańsk","Wrocław","Poznań","Łódź","Katowice","Lublin","Białystok","Szczecin"],
    },
    "Kazakhstan": {
        "first": [
            "Arman","Nursultan","Daniyar","Dias","Timur","Azamat","Erlan","Alikhan","Miras","Bekzat",
            "Serik","Aibek","Aslan","Sanzhar","Rustam","Ilyas","Madi","Adil","Yerkebulan","Nurali",
            "Islam","Zhandos","Eldar","Marat","Kanat","Aset","Bolat","Yerlan","Samat","Anuar",
        ],
        "last": [
            "Sadykov","Nurgaliyev","Akhmetov","Omarov","Kassymov","Zhaksylykov","Beketov","Zhumabayev","Tulegenov","Kenzhebayev",
            "Isayev","Zhanuzakov","Zhakipov","Rakhimov","Kalimullin","Suleimenov","Mukhamedov","Abdullayev","Saparov","Baiburin",
            "Kucherov","Mamin","Yesenov","Zhanbyr","Aryn","Kairatov","Sarsenov","Altynbek","Yessimov","Temirbayev",
        ],
        "towns": ["Astana","Almaty","Karaganda","Shymkent","Pavlodar","Ust-Kamenogorsk","Aktobe","Kostanay","Atyrau","Kokshetau"],
    },
    "Japan": {
        "first": [
            "Haruto","Yuto","Sota","Yuki","Kaito","Ryota","Daiki","Sho","Ren","Takumi",
            "Taiga","Hikaru","Kenta","Riku","Tsubasa","Yamato","Kohei","Kazuki","Shun","Takeru",
            "Keisuke","Masato","Naoki","Tatsuya","Hiroto","Sora","Ryo","Kenji","Jun","Akira",
        ],
        "last": [
            "Sato","Suzuki","Takahashi","Tanaka","Watanabe","Ito","Yamamoto","Nakamura","Kobayashi","Kato",
            "Yoshida","Yamada","Sasaki","Yamaguchi","Matsumoto","Inoue","Kimura","Hayashi","Shimizu","Yamazaki",
            "Mori","Abe","Ikeda","Hashimoto","Ishikawa","Nakajima","Maeda","Fujita","Okada","Hasegawa",
        ],
        "towns": ["Sapporo","Tokyo","Yokohama","Nagoya","Osaka","Sendai","Fukuoka","Kobe","Niigata","Hakodate"],
    },
    "South Korea": {
        "first": [
            "Minjun","Seo-jun","Do-yoon","Ji-ho","Ha-joon","Joon-woo","Jun-seo","Hyun-woo","Woo-jin","Ji-hun",
            "Sung-min","Tae-hyun","Dong-hyun","Seung-hyun","Yoon-ho","Jae-hyun","Jin-woo","Kang-min","Young-ho","Hyeon-jun",
            "Kyung-ho","Jae-min","In-woo","Seung-woo","Hyun-jun","Joon-ho","Sung-ho","Tae-yang","Soo-bin","Chan-woo",
        ],
        "last": [
            "Kim","Lee","Park","Choi","Jung","Kang","Cho","Yoon","Jang","Lim",
            "Han","Oh","Seo","Shin","Kwon","Hwang","Ahn","Song","Jeon","Hong",
            "Yang","Moon","Baek","Ryu","Son","Ko","Nam","Jin","Heo","Yu",
        ],
        "towns": ["Seoul","Busan","Incheon","Daegu","Daejeon","Gwangju","Suwon","Ulsan","Seongnam","Goyang"],
    },
    "China": {
        "first": [
            "Wei","Jun","Hao","Tao","Lei","Jie","Ming","Chen","Yang","Qiang",
            "Peng","Bo","Rui","Yong","Xin","Zhi","Yu","Xiang","Guo","Shan",
            "Yichen","Zihan","Haoran","Yuze","Zeyu","Cheng","Hanyu","Tianyu","Yichen","Jiahao",
        ],
        "last": [
            "Wang","Li","Zhang","Liu","Chen","Yang","Huang","Zhao","Wu","Zhou",
            "Xu","Sun","Ma","Zhu","Hu","Guo","He","Gao","Lin","Luo",
            "Zheng","Liang","Xie","Song","Tang","Han","Feng","Yu","Dong","Xiao",
        ],
        "towns": ["Beijing","Shanghai","Harbin","Shenyang","Dalian","Tianjin","Chengdu","Wuhan","Hangzhou","Shenzhen"],
    },
    "Australia": {
        "first": [
            "Jack","Noah","Oliver","William","Thomas","James","Liam","Lucas","Henry","Charlie",
            "Ethan","Benjamin","Alexander","Harrison","Leo","Oscar","Joshua","Samuel","Max","Ryan",
            "Cooper","Zachary","Daniel","Matthew","Patrick","Connor","Caleb","Nathan","Aiden","Jake",
        ],
        "last": [
            "Smith","Jones","Williams","Brown","Taylor","Wilson","Johnson","White","Martin","Anderson",
            "Thompson","Lee","Walker","Hall","Harris","Lewis","Young","King","Wright","Scott",
            "Cooper","Evans","Morgan","Turner","Parker","Bennett","Reid","Murray","Clark","Baker",
        ],
        "towns": ["Sydney","Melbourne","Brisbane","Perth","Adelaide","Canberra","Hobart","Darwin","Newcastle","Gold Coast"],
    },
    "New Zealand": {
        "first": [
            "Jack","Oliver","Noah","Liam","James","William","Lucas","Thomas","Henry","Charlie",
            "Leo","Ethan","Benjamin","Alexander","Samuel","Daniel","Max","Ryan","Finn","George",
            "Oscar","Joshua","Caleb","Nathan","Jacob","Matthew","Luke","Connor","Aiden","Jake",
        ],
        "last": [
            "Smith","Jones","Williams","Brown","Taylor","Wilson","Johnson","White","Thompson","Martin",
            "Anderson","Walker","Hall","Harris","Lewis","Young","King","Wright","Scott","Murray",
            "Cooper","Evans","Morgan","Turner","Parker","Bennett","Reid","Clark","Baker","Mitchell",
        ],
        "towns": ["Auckland","Wellington","Christchurch","Hamilton","Dunedin","Tauranga","Napier","Nelson","Palmerston North","Rotorua"],
    },
    "Brazil": {
        "first": [
            "Gabriel","Lucas","Mateus","Pedro","Guilherme","Gustavo","Rafael","Bruno","Felipe","Joao",
            "Enzo","Arthur","Daniel","Leonardo","Thiago","Caio","Victor","Rodrigo","Vitor","Igor",
            "Diego","André","Henrique","Matheus","Eduardo","Marcelo","Renan","Samuel","Vinicius","Luiz",
        ],
        "last": [
            "Silva","Santos","Oliveira","Souza","Rodrigues","Ferreira","Alves","Pereira","Lima","Gomes",
            "Costa","Ribeiro","Carvalho","Araujo","Martins","Rocha","Barbosa","Melo","Correia","Dias",
            "Teixeira","Moreira","Cardoso","Campos","Freitas","Ramos","Nunes","Mendes","Vieira","Monteiro",
        ],
        "towns": ["São Paulo","Rio de Janeiro","Belo Horizonte","Curitiba","Porto Alegre","Brasília","Florianópolis","Recife","Salvador","Campinas"],
    },
    "Argentina": {
        "first": [
            "Mateo","Juan","Santiago","Nicolas","Tomas","Lucas","Benjamin","Franco","Martin","Joaquin",
            "Agustin","Facundo","Bruno","Thiago","Gonzalo","Ignacio","Lautaro","Alejandro","Diego","Emiliano",
            "Rodrigo","Julian","Pablo","Marcos","Ezequiel","Kevin","Adrian","Federico","Hernan","Ricardo",
        ],
        "last": [
            "Gonzalez","Rodriguez","Fernandez","Lopez","Martinez","Garcia","Perez","Sanchez","Romero","Diaz",
            "Alvarez","Torres","Ruiz","Flores","Acosta","Benitez","Medina","Herrera","Suarez","Castro",
            "Ortiz","Vega","Rojas","Molina","Morales","Silva","Aguirre","Gutierrez","Juarez","Dominguez",
        ],
        "towns": ["Buenos Aires","Córdoba","Rosario","Mendoza","La Plata","Mar del Plata","San Miguel de Tucumán","Salta","Santa Fe","Neuquén"],
    },
    "Mexico": {
        "first": [
            "Santiago","Mateo","Sebastian","Leonardo","Diego","Daniel","Alejandro","Emiliano","Gabriel","Adrian",
            "David","Jose","Miguel","Javier","Fernando","Angel","Carlos","Juan","Luis","Rodrigo",
            "Jorge","Rafael","Hector","Marco","Pedro","Pablo","Raul","Ricardo","Victor","Andres",
        ],
        "last": [
            "Garcia","Hernandez","Martinez","Lopez","Gonzalez","Perez","Rodriguez","Sanchez","Ramirez","Cruz",
            "Flores","Gomez","Morales","Vazquez","Reyes","Jimenez","Torres","Diaz","Gutierrez","Ruiz",
            "Mendoza","Ortiz","Castillo","Alvarez","Romero","Chavez","Rivera","Ramos","Silva","Soto",
        ],
        "towns": ["Mexico City","Guadalajara","Monterrey","Puebla","Tijuana","León","Querétaro","Mérida","Toluca","Chihuahua"],
    },
    "Nigeria": {
        "first": [
            "Chinedu","Emeka","Ifeanyi","Kelechi","Uche","Chukwuemeka","Seyi","Tunde","Kunle","Adeyemi",
            "Ibrahim","Musa","Sani","Abdul","Umar","Samuel","David","Joseph","Daniel","Peter",
            "Chukwu","Obinna","Tochukwu","Nnamdi","Oluwaseun","Oluwatobi","Ayodele","Babatunde","Chisom","Ebuka",
        ],
        "last": [
            "Okafor","Nwankwo","Okoye","Eze","Ogunleye","Adebayo","Okonkwo","Ibrahim","Abubakar","Balogun",
            "Adesina","Olawale","Ojo","Akinyemi","Akinola","Nnamdi","Chukwu","Onyeka","Nwachukwu","Iheanacho",
            "Oladipo","Okechukwu","Ekwueme","Umeh","Nwosu","Ogunbiyi","Adekunle","Adeleke","Mohammed","Suleiman",
        ],
        "towns": ["Lagos","Abuja","Ibadan","Kano","Port Harcourt","Kaduna","Enugu","Benin City","Jos","Aba"],
    },
    "Kenya": {
        "first": [
            "Brian","Kevin","Eric","Dennis","John","James","Peter","Daniel","Samuel","Joseph",
            "Michael","David","Charles","Patrick","Anthony","George","Kennedy","Vincent","Paul","Stephen",
            "Mark","Victor","Allan","Martin","Nicholas","Francis","Collins","Emmanuel","Philip","Andrew",
        ],
        "last": [
            "Mwangi","Wanjiku","Otieno","Ochieng","Kamau","Kipchoge","Kiplagat","Njoroge","Wambui","Kariuki",
            "Mutua","Ndungu","Maina","Githinji","Njenga","Muriithi","Kibet","Korir","Cheruiyot","Chebet",
            "Koech","Rono","Chege","Karanja","Okello","Barasa","Wekesa","Omondi","Wafula","Musyoka",
        ],
        "towns": ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret","Thika","Malindi","Kitale","Nyeri","Machakos"],
    },
    "South Africa": {
        "first": [
            "Liam","Ethan","Noah","Joshua","Luke","Daniel","Michael","Ryan","Kyle","Matthew",
            "James","Tyler","Jordan","Connor","Jason","Dean","Ruben","Marco","Siyabonga","Thabo",
            "Kagiso","Sipho","Bongani","Kevin","Brandon","Dylan","Nathan","Cameron","Tristan","Justin",
        ],
        "last": [
            "van der Merwe","Smith","Botha","Naidoo","Nkosi","Dlamini","Ndlovu","Joubert","Du Plessis","Pretorius",
            "Mokoena","Mthembu","Pillay","Williams","Jacobs","Fourie","Nel","Coetzee","Kruger","Swanepoel",
            "Mabaso","Khumalo","Mahlangu","Govender","Venter","Muller","Marais","Barnard","Steyn","Zulu",
        ],
        "towns": ["Johannesburg","Cape Town","Durban","Pretoria","Port Elizabeth","Bloemfontein","Polokwane","Nelspruit","Kimberley","East London"],
    },
    "India": {
        "first": [
            "Arjun","Rohan","Aarav","Vihaan","Aditya","Ishaan","Karan","Rahul","Aman","Siddharth",
            "Vivek","Ankit","Ravi","Manish","Akash","Nikhil","Varun","Sanjay","Pranav","Kabir",
            "Dev","Harsh","Neeraj","Gaurav","Vikram","Suraj","Ajay","Arnav","Yash","Mohit",
        ],
        "last": [
            "Sharma","Verma","Gupta","Singh","Kumar","Patel","Mehta","Reddy","Nair","Iyer",
            "Chatterjee","Banerjee","Das","Jain","Joshi","Malhotra","Kapoor","Bose","Saxena","Aggarwal",
            "Bhat","Khan","Ali","Mukherjee","Pandey","Thakur","Yadav","Goswami","Sethi","Rao",
        ],
        "towns": ["Delhi","Mumbai","Bengaluru","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad","Jaipur","Chandigarh"],
    },
    "Philippines": {
        "first": [
            "Juan","Jose","Miguel","Angelo","Mark","John","Christian","Joshua","Gabriel","Daniel",
            "Paolo","Carlo","Rafael","Nathan","Enzo","Luis","Marco","Andre","Ryan","Kevin",
            "Jerome","Jayson","Francis","Vincent","Albert","Eduardo","Emmanuel","Noel","Ricardo","Santiago",
        ],
        "last": [
            "Santos","Reyes","Cruz","Bautista","Garcia","Mendoza","Torres","Flores","Ramos","Gonzales",
            "Rivera","Gomez","Castillo","Lopez","Hernandez","Perez","Domingo","Villanueva","Navarro","Aquino",
            "Dela Cruz","Manalo","De Leon","Salazar","Valdez","Marquez","Pascual","Rosales","Estrada","Padilla",
        ],
        "towns": ["Manila","Quezon City","Cebu City","Davao City","Baguio","Iloilo City","Cagayan de Oro","Bacolod","Zamboanga City","Pasig"],
    },
}


_NICKNAMES = [
    "Ace","Rocket","Hammer","Ghost","Moose","Doc","Skates","Turbo","Sparks","Tank",
    "Iceman","Razor","Chippy","Chief","Snipe","Wheels","Brick","Buzz","Professor","Flash",
]


def _nickname_from_name(rng, first: str, last: str) -> str:
    if rng.random() < 0.55:
        return _safe_choice(rng, _NICKNAMES)
    if rng.random() < 0.5:
        return first[:1] + last[:1]
    return first


def _pronunciation_key(nationality: str, full_name: str) -> Optional[str]:
    # Optional and intentionally light; can be expanded later.
    nat = nationality.lower()
    if nat in ("finland", "sweden", "czechia", "slovakia"):
        return full_name.replace("ä", "ae").replace("ö", "oe").replace("å", "o")
    if nat in ("russia", "ukraine", "belarus", "kazakhstan"):
        return full_name
    return None


def choose_nationality(rng, *, market_bias: Optional[Dict[str, float]] = None) -> str:
    """
    Select nationality with an NHL-ish distribution but includes non-traditional markets.
    `market_bias` can override/shift weights (deterministic given rng).
    """
    base = {
        "Canada": 0.36,
        "USA": 0.22,
        "Sweden": 0.08,
        "Finland": 0.07,
        "Russia": 0.07,
        "Czechia": 0.05,
        "Slovakia": 0.03,
        "Germany": 0.03,
        "Switzerland": 0.02,
        "France": 0.01,
        "UK": 0.01,
        "Norway": 0.01,
        "Denmark": 0.01,
        "Latvia": 0.01,
        "Belarus": 0.005,
        "Ukraine": 0.005,
        "Poland": 0.005,
        "Kazakhstan": 0.005,
        "Japan": 0.003,
        "South Korea": 0.003,
        "China": 0.003,
        "Australia": 0.003,
        "New Zealand": 0.001,
        "Brazil": 0.002,
        "Argentina": 0.002,
        "Mexico": 0.002,
        "Nigeria": 0.002,
        "Kenya": 0.001,
        "South Africa": 0.002,
        "India": 0.002,
        "Philippines": 0.002,
    }
    if market_bias:
        for k, v in market_bias.items():
            if k in base:
                base[k] = max(0.0, float(base[k]) + float(v))

    # normalize and draw
    items = [(k, w) for k, w in base.items() if w > 0.0 and k in NAME_POOLS]
    total = sum(w for _, w in items) or 1.0
    roll = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if roll <= acc:
            return k
    return items[-1][0]


def generate_human_identity(rng, *, nationality: Optional[str] = None) -> HumanIdentity:
    nat = nationality or choose_nationality(rng)
    pool = NAME_POOLS.get(nat) or NAME_POOLS["Canada"]
    first = _safe_choice(rng, pool.get("first", []))
    last = _safe_choice(rng, pool.get("last", []))
    hometown = _safe_choice(rng, pool.get("towns", []))
    full = f"{first} {last}"

    nickname = None
    if rng.random() < 0.14:
        nickname = _nickname_from_name(rng, first, last)

    pron = _pronunciation_key(nat, full)
    return HumanIdentity(full_name=full, nationality=nat, hometown=hometown, nickname=nickname, pronunciation=pron)


# Backwards-compatible: keep generate_name for older callers
def generate_name(rng) -> str:
    return generate_human_identity(rng).full_name

