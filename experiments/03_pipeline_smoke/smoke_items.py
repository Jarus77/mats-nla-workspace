"""Toy in-J feature for the pipeline smoke: language identity, French vs German.

40 content-matched pairs (same meaning, one French and one German sentence), so items cluster
by pair. Plain text (no chat template); read position = last token. Feature value a = French
(+), b = German (-).
"""

PAIRS: list[tuple[str, str]] = [
    ("Le chat dort tranquillement sur le canapé du salon.", "Die Katze schläft ruhig auf dem Sofa im Wohnzimmer."),
    ("Nous allons au marché chaque samedi matin.", "Wir gehen jeden Samstagmorgen auf den Markt."),
    ("Mon frère travaille dans une petite boulangerie.", "Mein Bruder arbeitet in einer kleinen Bäckerei."),
    ("Il pleut depuis trois jours sans arrêt.", "Es regnet seit drei Tagen ohne Pause."),
    ("Elle a oublié ses clés dans la voiture.", "Sie hat ihre Schlüssel im Auto vergessen."),
    ("Les enfants jouent au football dans le parc.", "Die Kinder spielen Fußball im Park."),
    ("Je voudrais une tasse de café, s'il vous plaît.", "Ich hätte gern eine Tasse Kaffee, bitte."),
    ("Le train pour Lyon part à huit heures.", "Der Zug nach Lyon fährt um acht Uhr ab."),
    ("Ma grand-mère prépare une soupe aux légumes.", "Meine Großmutter kocht eine Gemüsesuppe."),
    ("Nous avons visité un vieux château hier.", "Wir haben gestern eine alte Burg besucht."),
    ("Le médecin m'a conseillé de dormir davantage.", "Der Arzt hat mir geraten, mehr zu schlafen."),
    ("Ils ont acheté une nouvelle maison à la campagne.", "Sie haben ein neues Haus auf dem Land gekauft."),
    ("La bibliothèque ferme tôt le dimanche.", "Die Bibliothek schließt am Sonntag früh."),
    ("Mon ordinateur ne s'allume plus depuis ce matin.", "Mein Computer lässt sich seit heute Morgen nicht mehr einschalten."),
    ("Le professeur explique la leçon avec patience.", "Der Lehrer erklärt die Lektion mit Geduld."),
    ("Nous devons finir ce projet avant vendredi.", "Wir müssen dieses Projekt vor Freitag abschließen."),
    ("Les fleurs du jardin sentent très bon.", "Die Blumen im Garten riechen sehr gut."),
    ("Il a couru dix kilomètres ce matin.", "Er ist heute Morgen zehn Kilometer gelaufen."),
    ("La réunion a été reportée à la semaine prochaine.", "Die Besprechung wurde auf nächste Woche verschoben."),
    ("Je lis un roman policier très captivant.", "Ich lese einen sehr spannenden Kriminalroman."),
    ("Le magasin est fermé pour travaux.", "Das Geschäft ist wegen Umbauarbeiten geschlossen."),
    ("Ma sœur apprend à jouer du piano.", "Meine Schwester lernt Klavier spielen."),
    ("Nous avons mangé dans un restaurant italien.", "Wir haben in einem italienischen Restaurant gegessen."),
    ("Le vent souffle fort sur la côte.", "Der Wind weht stark an der Küste."),
    ("Ils regardent un film à la télévision.", "Sie sehen sich einen Film im Fernsehen an."),
    ("La voisine promène son chien tous les soirs.", "Die Nachbarin geht jeden Abend mit ihrem Hund spazieren."),
    ("J'ai perdu mon parapluie dans le métro.", "Ich habe meinen Regenschirm in der U-Bahn verloren."),
    ("Le gouvernement a annoncé de nouvelles mesures.", "Die Regierung hat neue Maßnahmen angekündigt."),
    ("Les oiseaux chantent dès le lever du soleil.", "Die Vögel singen schon bei Sonnenaufgang."),
    ("Nous partons en vacances à la montagne.", "Wir fahren in den Urlaub in die Berge."),
    ("Mon père répare le vélo dans le garage.", "Mein Vater repariert das Fahrrad in der Garage."),
    ("Cette entreprise fabrique des meubles en bois.", "Diese Firma stellt Möbel aus Holz her."),
    ("Le concert commence dans une demi-heure.", "Das Konzert beginnt in einer halben Stunde."),
    ("Elle écrit une lettre à son ancien professeur.", "Sie schreibt einen Brief an ihren ehemaligen Lehrer."),
    ("Les prix ont beaucoup augmenté cette année.", "Die Preise sind dieses Jahr stark gestiegen."),
    ("Il faut arroser les plantes deux fois par semaine.", "Man muss die Pflanzen zweimal pro Woche gießen."),
    ("Le musée présente une exposition sur la mer.", "Das Museum zeigt eine Ausstellung über das Meer."),
    ("Nous attendons le bus depuis vingt minutes.", "Wir warten seit zwanzig Minuten auf den Bus."),
    ("Mon collègue parle trois langues couramment.", "Mein Kollege spricht fließend drei Sprachen."),
    ("La neige recouvre les toits de la ville.", "Der Schnee bedeckt die Dächer der Stadt."),
]

LANG_A, LANG_B = "French", "German"

# Native-format candidate pairs (differ only in the feature value), frozen for the smoke.
NLA_PAIRS = [
    ("French-language text", "German-language text"),
    ("A short everyday sentence written in French", "A short everyday sentence written in German"),
    ("The text is in French", "The text is in German"),
]
OL_PAIRS = [
    ("- French language", "- German language"),
    ("- French", "- German"),
    ("- written in French", "- written in German"),
]
JLENS_WORDS = (["French", "français", "France"], ["German", "Deutsch", "Germany"])
GEMINI_QUESTION = "Which language is the text written in?"
GEMINI_EXPL_QUESTION = ("This is a description of a language model's internal state while it "
                        "read some text. Which language was that text written in?")


def items() -> list[dict]:
    """[{text, label (1 = French), pair}] in pair order (French first)."""
    out = []
    for k, (fr, de) in enumerate(PAIRS):
        out.append({"text": fr, "label": 1, "pair": k})
        out.append({"text": de, "label": 0, "pair": k})
    return out
