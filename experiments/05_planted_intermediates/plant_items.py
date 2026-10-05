"""Items for (C) planted intermediates: two-hop prompts whose bridge is a COUNTRY.

Two sources:
- ``bank``: the WorkspaceBench multihop items whose bridge is a country in TABLE and whose
  relation is one of RELATIONS (30 of the 100; each mapped by hand below).
- ``cons``: a constructed bank, country x clue x relation, "Fact: The <relation> of the country
  famous for <clue> is" (5 clues per country). Its role is (a) more items and (b) paraphrase-
  diverse prompts per bridge, from which the bridge feature directions are built
  (class means over clues/relations, leave-the-item's-clue-out).

Answers carry aliases; an answer is "produced" if some alias is greedy-consistent (every alias
token is the argmax given the prompt and the alias prefix).
"""

from __future__ import annotations

import json
from pathlib import Path

RELATIONS = ("capital", "language", "currency", "continent")
TEMPLATES = {
    "capital": "Fact: The capital of the country famous for {clue} is",
    "language": "Fact: The official language of the country famous for {clue} is",
    "currency": "Fact: The currency of the country famous for {clue} is the",
    "continent": "Fact: The continent of the country famous for {clue} is",
}

# code: (name, J-Lens single-token name variants, clues, {relation: aliases or None (skipped)})
TABLE: dict[str, dict] = {
    "FR": dict(name="France", jl=["France", "French"], clues=["the Eiffel Tower", "the Louvre museum", "croissants and baguettes", "the Palace of Versailles", "Mont Saint-Michel"],
               ans=dict(capital=["Paris"], language=["French"], currency=["euro"], continent=["Europe"])),
    "IT": dict(name="Italy", jl=["Italy", "Italian"], clues=["the Colosseum", "the Leaning Tower of Pisa", "the canals and gondolas of Venice", "pizza and pasta", "the ruins of Pompeii"],
               ans=dict(capital=["Rome"], language=["Italian"], currency=["euro"], continent=["Europe"])),
    "JP": dict(name="Japan", jl=["Japan", "Japanese"], clues=["Mount Fuji", "sushi", "samurai warriors", "cherry blossoms and geishas", "the Shinkansen bullet train"],
               ans=dict(capital=["Tokyo"], language=["Japanese"], currency=["yen", "Japanese yen"], continent=["Asia"])),
    "EG": dict(name="Egypt", jl=["Egypt", "Egyptian"], clues=["the pyramids of Giza", "the Great Sphinx", "the pharaohs", "the tomb of Tutankhamun", "the Suez Canal"],
               ans=dict(capital=["Cairo"], language=["Arabic"], currency=["Egyptian pound"], continent=["Africa"])),
    "ES": dict(name="Spain", jl=["Spain", "Spanish"], clues=["flamenco dancing", "the Alhambra palace", "paella", "the Sagrada Familia", "the running of the bulls in Pamplona"],
               ans=dict(capital=["Madrid"], language=["Spanish"], currency=["euro"], continent=["Europe"])),
    "GR": dict(name="Greece", jl=["Greece", "Greek"], clues=["the Acropolis", "the Parthenon", "Mount Olympus", "the island of Santorini", "the ancient Olympic Games"],
               ans=dict(capital=["Athens"], language=["Greek"], currency=["euro"], continent=["Europe"])),
    "BR": dict(name="Brazil", jl=["Brazil", "Brazilian"], clues=["the Christ the Redeemer statue", "samba and Carnival", "Copacabana beach", "the footballer Pelé", "the Amazon rainforest"],
               ans=dict(capital=["Brasília", "Brasilia"], language=["Portuguese"], currency=["real", "Brazilian real"], continent=["South America"])),
    "CN": dict(name="China", jl=["China", "Chinese"], clues=["the Great Wall", "the Forbidden City", "giant pandas", "the Terracotta Army", "the Yangtze River"],
               ans=dict(capital=["Beijing"], language=["Mandarin", "Chinese", "Mandarin Chinese"], currency=["yuan", "renminbi", "Chinese yuan"], continent=["Asia"])),
    "IN": dict(name="India", jl=["India", "Indian"], clues=["the Taj Mahal", "Bollywood films", "the sacred Ganges river", "the leader Mahatma Gandhi", "the Golden Temple of Amritsar"],
               ans=dict(capital=["New Delhi", "Delhi"], language=["Hindi"], currency=["rupee", "Indian rupee"], continent=["Asia"])),
    "RU": dict(name="Russia", jl=["Russia", "Russian"], clues=["the Kremlin", "Red Square", "matryoshka dolls", "the Bolshoi Ballet", "the Trans-Siberian Railway"],
               ans=dict(capital=["Moscow"], language=["Russian"], currency=["ruble", "rouble", "Russian ruble"], continent=None)),
    "AU": dict(name="Australia", jl=["Australia", "Australian"], clues=["kangaroos", "the Sydney Opera House", "the Great Barrier Reef", "Uluru", "koalas"],
               ans=dict(capital=["Canberra"], language=["English"], currency=["Australian dollar"], continent=None)),
    "CA": dict(name="Canada", jl=["Canada", "Canadian"], clues=["the red maple leaf flag", "the CN Tower in Toronto", "Banff National Park", "poutine", "ice hockey and the Mounties"],
               ans=dict(capital=["Ottawa"], language=None, currency=["Canadian dollar"], continent=["North America"])),
    "DE": dict(name="Germany", jl=["Germany", "German"], clues=["Oktoberfest", "the Brandenburg Gate", "Neuschwanstein Castle", "the Autobahn", "the Black Forest"],
               ans=dict(capital=["Berlin"], language=["German"], currency=["euro"], continent=["Europe"])),
    "MX": dict(name="Mexico", jl=["Mexico", "Mexican"], clues=["the pyramid of Chichen Itza", "the ancient Aztecs", "tacos and tequila", "the Day of the Dead festival", "mariachi music"],
               ans=dict(capital=["Mexico City"], language=["Spanish"], currency=["peso", "Mexican peso"], continent=["North America"])),
    "NO": dict(name="Norway", jl=["Norway", "Norwegian"], clues=["its fjords", "the painting The Scream", "the Lofoten islands", "the Nobel Peace Prize ceremony", "salmon farming and North Sea oil"],
               ans=dict(capital=["Oslo"], language=["Norwegian"], currency=["Norwegian krone", "krone"], continent=["Europe"])),
    "SE": dict(name="Sweden", jl=["Sweden", "Swedish"], clues=["IKEA", "the pop group ABBA", "Volvo cars", "the Nobel Prize banquet", "the Icehotel in Jukkasjärvi"],
               ans=dict(capital=["Stockholm"], language=["Swedish"], currency=["Swedish krona", "krona"], continent=["Europe"])),
    "NL": dict(name="Netherlands", jl=["Netherlands", "Dutch", "Holland"], clues=["tulips and windmills", "wooden clogs", "Gouda cheese", "the Anne Frank House", "the painter Rembrandt"],
               ans=dict(capital=["Amsterdam"], language=["Dutch"], currency=["euro"], continent=["Europe"])),
    "IE": dict(name="Ireland", jl=["Ireland", "Irish"], clues=["Guinness", "the shamrock", "the Cliffs of Moher", "leprechauns", "Saint Patrick"],
               ans=dict(capital=["Dublin"], language=None, currency=["euro"], continent=["Europe"])),
    "PT": dict(name="Portugal", jl=["Portugal", "Portuguese"], clues=["port wine", "fado music", "the explorer Vasco da Gama", "pastel de nata custard tarts", "the footballer Cristiano Ronaldo"],
               ans=dict(capital=["Lisbon"], language=["Portuguese"], currency=["euro"], continent=["Europe"])),
    "AR": dict(name="Argentina", jl=["Argentina", "Argentine"], clues=["the tango", "the footballer Lionel Messi", "the gauchos of the Pampas", "Eva Perón", "Malbec wine from Mendoza"],
               ans=dict(capital=["Buenos Aires"], language=["Spanish"], currency=["peso", "Argentine peso"], continent=["South America"])),
    "CH": dict(name="Switzerland", jl=["Switzerland", "Swiss"], clues=["cuckoo clocks", "the Matterhorn", "neutrality and private banking", "Toblerone chocolate", "luxury watchmaking"],
               ans=dict(capital=["Bern"], language=None, currency=["Swiss franc", "franc"], continent=["Europe"])),
    "TR": dict(name="Turkey", jl=["Turkey", "Turkish"], clues=["the Hagia Sophia", "the hot-air balloons of Cappadocia", "the Blue Mosque", "the ancient city of Troy", "the Bosphorus strait"],
               ans=dict(capital=["Ankara"], language=["Turkish"], currency=["Turkish lira", "lira"], continent=None)),
    "KR": dict(name="South Korea", jl=["Korea", "Korean"], clues=["K-pop", "kimchi", "Samsung", "the song Gangnam Style", "taekwondo"],
               ans=dict(capital=["Seoul"], language=["Korean"], currency=["won", "South Korean won"], continent=["Asia"])),
    "PE": dict(name="Peru", jl=["Peru", "Peruvian"], clues=["Machu Picchu", "the Inca Empire", "the Nazca Lines", "ceviche", "the city of Cusco"],
               ans=dict(capital=["Lima"], language=["Spanish"], currency=["sol", "Peruvian sol"], continent=["South America"])),
    "GB": dict(name="United Kingdom", jl=["Britain", "British", "UK", "England"], clues=["Big Ben", "Stonehenge", "Buckingham Palace", "red double-decker buses", "William Shakespeare"],
               ans=dict(capital=["London"], language=["English"], currency=["pound sterling", "pound", "British pound"], continent=["Europe"])),
    "US": dict(name="United States", jl=["America", "American", "USA"], clues=["the Statue of Liberty", "the Grand Canyon", "Hollywood", "Mount Rushmore", "the Super Bowl"],
               ans=dict(capital=["Washington, D.C.", "Washington"], language=["English"], currency=["US dollar", "dollar", "United States dollar"], continent=["North America"])),
    "DK": dict(name="Denmark", jl=["Denmark", "Danish"], clues=["LEGO", "Hans Christian Andersen", "the Little Mermaid statue", "Tivoli Gardens", "hygge"],
               ans=dict(capital=["Copenhagen"], language=["Danish"], currency=["Danish krone", "krone"], continent=["Europe"])),
    "FI": dict(name="Finland", jl=["Finland", "Finnish"], clues=["Nokia", "saunas", "the Moomins", "the composer Sibelius", "the video game Angry Birds"],
               ans=dict(capital=["Helsinki"], language=["Finnish"], currency=["euro"], continent=["Europe"])),
    "PL": dict(name="Poland", jl=["Poland", "Polish"], clues=["the composer Chopin", "the astronomer Copernicus", "pierogi dumplings", "the Wieliczka salt mine", "the Solidarity movement"],
               ans=dict(capital=["Warsaw"], language=["Polish"], currency=["złoty", "zloty"], continent=["Europe"])),
    "VN": dict(name="Vietnam", jl=["Vietnam", "Vietnamese"], clues=["Ha Long Bay", "pho noodle soup", "the Cu Chi tunnels", "conical hats and the ao dai", "the city of Hoi An"],
               ans=dict(capital=["Hanoi"], language=["Vietnamese"], currency=["dong", "Vietnamese dong"], continent=["Asia"])),
    "KE": dict(name="Kenya", jl=["Kenya", "Kenyan"], clues=["the Maasai Mara reserve", "the Maasai people", "the marathon runner Eliud Kipchoge", "safaris in Amboseli", "Lake Turkana"],
               ans=dict(capital=["Nairobi"], language=None, currency=["Kenyan shilling", "shilling"], continent=["Africa"])),
    "MA": dict(name="Morocco", jl=["Morocco", "Moroccan"], clues=["the souks of Marrakech", "couscous and tagine", "the Hassan II Mosque", "the city of Casablanca", "the blue city of Chefchaouen"],
               ans=dict(capital=["Rabat"], language=["Arabic"], currency=["dirham", "Moroccan dirham"], continent=["Africa"])),
}

# WorkspaceBench multihop items with a country bridge in TABLE and a relation in RELATIONS
BANK_MAP = {
    "bf-boot-capital": ("IT", "capital"), "bf-eiffel-currency": ("FR", "currency"),
    "bf-giza-continent": ("EG", "continent"), "bf-madrid-language": ("ES", "language"),
    "bf-rio2016-language": ("BR", "language"), "bf-sushi-capital": ("JP", "capital"),
    "b3-mh-acropolis-capital": ("GR", "capital"), "b3-mh-buenosaires-language": ("AR", "language"),
    "b3-mh-colosseum-capital": ("IT", "capital"), "b3-mh-eiffel-capital": ("FR", "capital"),
    "b3-mh-fjords-capital": ("NO", "capital"), "b3-mh-fuji-currency": ("JP", "currency"),
    "b3-mh-guinness-capital": ("IE", "capital"), "b3-mh-kremlin-language": ("RU", "language"),
    "b3-mh-lisbon-language": ("PT", "language"), "b3-mh-nokia-capital": ("FI", "capital"),
    "b3-mh-sphinx-capital": ("EG", "capital"), "b3-mh-tulips-capital": ("NL", "capital"),
    "bf-eiffel-continent": ("FR", "continent"), "bf-kangaroo-capital": ("AU", "capital"),
    "bf-mapleleaf-capital": ("CA", "capital"), "bf-olympia-language": ("GR", "language"),
    "b3-mh-ikea-capital": ("SE", "capital"), "b3-mh-lego-capital": ("DK", "capital"),
    "b3-mh-pizza-continent": ("IT", "continent"), "colosseum-currency": ("IT", "currency"),
    "bf-colosseum-language": ("IT", "language"), "bf-greatwall-language": ("CN", "language"),
    "bf-tajmahal-continent": ("IN", "continent"), "b3-mh-aztec-language": ("MX", "language"),
}

NEUTRAL_PROMPTS = [
    "The weather forecast for tomorrow says that it will be",
    "She opened the fridge and took out a bottle of",
    "def add(a, b):\n    return",
    "The meeting has been moved to next",
    "My favourite colour has always been",
    "After the long hike, we finally reached the",
]

_P = Path(__file__).resolve().parents
BANK_PATH = (_P[2] if len(_P) > 2 else _P[0]) / "third_party/workspace-bench/evals/multihop/items.json"


def build_items(bank_items: list[dict] | None = None) -> list[dict]:
    """All items: dicts with id, source, country, relation, clue (k or None), prompt,
    read_offset (negative), answers (aliases)."""
    if bank_items is None:
        bank_items = json.loads(BANK_PATH.read_text())["items"]
    out = []
    for c, row in TABLE.items():
        for rel in RELATIONS:
            if row["ans"].get(rel) is None:
                continue
            for k, clue in enumerate(row["clues"]):
                out.append(dict(id=f"cons-{c}-{rel}-{k}", source="cons", country=c, relation=rel,
                                clue=k, prompt=TEMPLATES[rel].format(clue=clue), read_offset=-1,
                                answers=_variants(row["ans"][rel])))
    by_name = {b["name"]: b for b in bank_items}
    for name, (c, rel) in BANK_MAP.items():
        b = by_name[name]
        out.append(dict(id=f"bank-{name}", source="bank", country=c, relation=rel, clue=None,
                        prompt=b["prompt"], read_offset=int(b["readout"]["offsets"][0]),
                        answers=_variants(TABLE[c]["ans"][rel])))
    return out


def _variants(aliases: list[str]) -> list[str]:
    """Aliases plus case variants (first letter upper, title case); order-preserving, unique."""
    out: list[str] = []
    for a in aliases:
        for v in (a, a[:1].upper() + a[1:], a.title()):
            if v not in out:
                out.append(v)
    return out


def answers_of(country: str, relation: str) -> list[str] | None:
    a = TABLE[country]["ans"].get(relation)
    return None if a is None else _variants(a)
