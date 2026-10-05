"""Feature specs for template mining: values, descriptions, split sizes and FIXED label templates
(the "4th column", D9 item 6). Pure data, no torch.

A template is ``{"id", "family", "template", "fillers"}``: ``template`` holds exactly one ``{v}``
slot; ``fillers`` maps each feature value key to the slot text, so the two filled candidates differ
ONLY in the feature value. For item-specific values (typo target) ``fillers == {"*": "{w}"}`` and
``{w}`` is replaced by the item's word (optionally transformed, e.g. ``{W}`` = capitalised).

Native formats (src/mats README): NLA = explanation register scored after ``<explanation>\\n``;
OL = a ``- `` bullet scored as the start of the response; J-Lens = label-token word lists.
"""

from __future__ import annotations

import re

N_MINE_PER_VALUE = 20      # mining split: 20 + 20 items per feature
N_SELECT_PER_VALUE = 80    # selection split: 80 + 80 items per feature
K_READOUTS = 8
OL_LAYERS = (42, 40, 44)   # 42 primary (D7); 40/44 sensitivity
NLA_LAYER = 42
N_SELECT_TEMPLATES = 3

# Count features -> two bins for template selection (scoring candidates are binary); the raw count
# is kept in item meta so a Spearman(LO, count) is also reported.
BINS = {
    "nj1_col": {"short": (10, 30), "long": (55, 85)},          # chars since last newline
    "bk_wordpos": {"early": (2, 5), "late": (18, 40)},         # words since sentence start
    "bk_listidx": {"early": (2, 3), "late": (8, 12)},          # 1-based bullet item index
    "bk_indent": {"shallow": (1, 1), "deep": (3, 4)},          # Python indentation level
    "ij_listnum": {"early": (2, 3), "late": (8, 12)},          # "Item N:" index (numerals in text)
}
# nj1_col_break (decision family): chars left to the wrap width W - col; col matched across classes.
BREAK_SOON_MAX, BREAK_FAR_MIN, BREAK_COL_RANGE = 6, 25, (40, 75)
# Features whose PRIMARY templates must be qualitative, numerals secondary (Fable round 4).
QUALITATIVE_PRIMARY = ("nj1_col", "nj1_col_break")

# Priority order (lead, 2026-10-05): col first, then the in-J calibrations, then backups.
FEATURES: dict[str, dict] = {
    "nj1_col": {
        "values": ["short", "long"],
        "desc": "characters since the last newline (current column) in hard-wrapped prose under "
                "automatic linewrap (wrap width 30-90), read at an arbitrary token inside a line. "
                "'short' = 10-30 characters into the line, 'long' = 55-85 characters into the line",
    },
    "nj1_col_break": {
        "values": ["soon", "far"],
        "desc": "whether a line break is coming under automatic linewrap (wrap width 30-90): "
                "'soon' = within 0-6 characters of the wrap width, 'far' = 25+ characters before "
                "it; the current column is matched (40-75) across both values, so this is the "
                "break DECISION rather than the raw count",
    },
    "ij2_lang": {
        "values": ["French", "German"],
        "desc": "language identity of the text (French vs German), read at the last token",
    },
    "nj2_tense": {
        "values": ["past", "present"],
        "desc": "verb tense of the current clause in natural English prose (past vs present), "
                "read at a token after the clause's finite verb",
    },
    "bk_indent": {
        "values": ["shallow", "deep"],
        "desc": "indentation level of the current line in Python code: 'shallow' = 1 level "
                "(4 spaces), 'deep' = 3-4 levels (12-16 spaces); read at a token inside the line",
    },
    "bk_wordpos": {
        "values": ["early", "late"],
        "desc": "number of words since the start of the current sentence in English prose: "
                "'early' = 2-5 words in, 'late' = 18-40 words in; read at the end of a word",
    },
    "bk_listidx": {
        "values": ["early", "late"],
        "desc": "index of the current item in an unnumbered '- ' bullet list: 'early' = item "
                "2-3, 'late' = item 8-12; read at the last token of that item",
    },
    "bk_caps": {
        "values": ["uppercase", "lowercase"],
        "desc": "capitalisation: the passage is written in ALL CAPITAL LETTERS ('uppercase') vs "
                "normal sentence case ('lowercase'); read at the last token",
    },
    "bk_code": {
        "values": ["Python", "JavaScript"],
        "desc": "programming language of a code snippet (Python vs JavaScript), read at the last "
                "token of a truncated snippet",
    },
    "ij_listnum": {
        "values": ["early", "late"],
        "desc": "index of the current item in a list whose items are labelled 'Item N:' (the "
                "numeral is visible in the text; in-J count control): 'early' = item 2-3, 'late' "
                "= item 8-12; read at the last token of that item",
    },
    "bk_quote": {
        "values": ["inside", "outside"],
        "desc": "in-quote state in English prose: the read token is 'inside' a double-quoted span "
                "(3+ words into it, before it closes) vs 'outside' any quote (4+ words after a "
                "closing quote); read at the end of a word",
    },
    "agr_gender": {
        "values": ["masculine", "feminine"],
        "desc": "grammatical gender of the UPCOMING noun in French or German text cut right "
                "after an article (French le/un/ce/du vs la/une/cette; German einen vs eine)",
    },
    "agr_number": {
        "values": ["singular", "plural"],
        "desc": "grammatical number of the UPCOMING noun in French text cut right after an "
                "article (le/la/un/une vs les/des)",
    },
    "nj3_predpos": {  # deprioritised (fails G2-geom); mined last because it is cheap
        "values": ["adjective", "noun"],
        "desc": "part of speech of the word the model predicts NEXT (adjective vs noun; ground "
                "truth = POS-tagged mass of the model's next-token distribution), read at the "
                "last prompt token",
    },
}

_COL = {"short": "twenty", "long": "seventy"}
# col fixed label = the lead's QUALITATIVE example (primary); the numeric label is kept as an extra
# non-selectable column ("fixed_numeric") because numerals are secondary for col (Fable round 4).
_COL_Q = {"short": "at the start of a line", "long": "near the end of a long line"}
FIXED_NUMERIC = {
    "nla": {"nj1_col": {"template": "The current line is about {v} characters long", "fillers": _COL}},
    "ol": {"nj1_col": {"template": "- {v} characters", "fillers": _COL}},
}

# Fixed label templates (not mined). NLA: explanation register; OL: '- ' bullet.
FIXED: dict[str, dict[str, dict]] = {
    "nla": {
        "nj1_col": {"template": "The current position is {v}", "fillers": _COL_Q},
        "nj1_col_break": {"template": "A line break is {v}",
                          "fillers": {"soon": "coming", "far": "not coming yet"}},
        "ij2_lang": {"template": "The text is in {v}",
                     "fillers": {"French": "French", "German": "German"}},
        "nj2_tense": {"template": "The verb is in the {v} tense",
                      "fillers": {"past": "past", "present": "present"}},
        "bk_indent": {"template": "The code is indented {v} deep",
                      "fillers": {"shallow": "one level", "deep": "three levels"}},
        "bk_wordpos": {"template": "The current sentence is {v} words long so far",
                       "fillers": {"early": "three", "late": "twenty"}},
        "bk_listidx": {"template": "This is item number {v} in the list",
                       "fillers": {"early": "two", "late": "nine"}},
        "bk_caps": {"template": "The text is written in {v} letters",
                    "fillers": {"uppercase": "uppercase", "lowercase": "lowercase"}},
        "bk_code": {"template": "The code is written in {v}",
                    "fillers": {"Python": "Python", "JavaScript": "JavaScript"}},
        "ij_listnum": {"template": "This is item number {v} in the list",
                       "fillers": {"early": "two", "late": "nine"}},
        "bk_quote": {"template": "The current text is {v} a quotation",
                     "fillers": {"inside": "inside", "outside": "outside"}},
        "agr_gender": {"template": "The next word is a {v} noun",
                       "fillers": {"masculine": "masculine", "feminine": "feminine"}},
        "agr_number": {"template": "The next word is a {v} noun",
                       "fillers": {"singular": "singular", "plural": "plural"}},
        "nj3_predpos": {"template": "The next word is likely {v}",
                        "fillers": {"adjective": "an adjective", "noun": "a noun"}},
    },
    "ol": {
        "nj1_col": {"template": "- {v}", "fillers": _COL_Q},
        "nj1_col_break": {"template": "- line break {v}", "fillers": {"soon": "coming", "far": "not coming yet"}},
        "ij2_lang": {"template": "- {v} language",
                     "fillers": {"French": "French", "German": "German"}},
        "nj2_tense": {"template": "- {v} tense", "fillers": {"past": "past", "present": "present"}},
        "bk_indent": {"template": "- {v} of indentation",
                      "fillers": {"shallow": "one level", "deep": "three levels"}},
        "bk_wordpos": {"template": "- {v} words", "fillers": {"early": "three", "late": "twenty"}},
        "bk_listidx": {"template": "- item {v}", "fillers": {"early": "two", "late": "nine"}},
        "bk_caps": {"template": "- {v}",
                    "fillers": {"uppercase": "uppercase", "lowercase": "lowercase"}},
        "bk_code": {"template": "- {v} code",
                    "fillers": {"Python": "Python", "JavaScript": "JavaScript"}},
        "ij_listnum": {"template": "- item {v}", "fillers": {"early": "two", "late": "nine"}},
        "bk_quote": {"template": "- {v} a quotation", "fillers": {"inside": "inside", "outside": "outside"}},
        "agr_gender": {"template": "- {v} noun", "fillers": {"masculine": "masculine", "feminine": "feminine"}},
        "agr_number": {"template": "- {v} noun", "fillers": {"singular": "singular", "plural": "plural"}},
        "nj3_predpos": {"template": "- {v}", "fillers": {"adjective": "adjective", "noun": "noun"}},
    },
}

# J-Lens fixed label words per value (single-token variants are taken by token_variants).
FIXED_JLENS: dict[str, dict[str, list[str]]] = {
    "nj1_col": {"short": ["twenty"], "long": ["seventy"]},
    "ij2_lang": {"French": ["French", "français", "France"], "German": ["German", "Deutsch", "Germany"]},
    "nj2_tense": {"past": ["past"], "present": ["present"]},
    "bk_indent": {"shallow": ["one", "single"], "deep": ["three", "deep"]},
    "bk_wordpos": {"early": ["three"], "late": ["twenty"]},
    "bk_listidx": {"early": ["two", "second"], "late": ["nine", "ninth"]},
    "bk_caps": {"uppercase": ["uppercase", "caps", "capital"], "lowercase": ["lowercase", "lower"]},
    "bk_code": {"Python": ["Python", "python"], "JavaScript": ["JavaScript", "javascript", "JS"]},
    "nj1_col_break": {"soon": ["newline", "break"], "far": ["continue", "middle"]},
    "ij_listnum": {"early": ["two", "second"], "late": ["nine", "ninth"]},
    "bk_quote": {"inside": ["inside", "quote"], "outside": ["outside"]},
    "agr_gender": {"masculine": ["masculine"], "feminine": ["feminine"]},
    "agr_number": {"singular": ["singular"], "plural": ["plural"]},
    "nj3_predpos": {"adjective": ["adjective", "adj"], "noun": ["noun"]},
}


_NUMWORD = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                      r"\w+teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
                      re.IGNORECASE)


def is_numeric(template: dict) -> bool:
    """True if any filled candidate carries a numeral / number word (secondary for col)."""
    return any(_NUMWORD.search(template["template"].replace("{v}", f)) for f in template["fillers"].values())


def fixed_template(reader: str, feature: str) -> dict:
    t = FIXED[reader][feature]
    return {"id": f"{reader}:{feature}:fixed", "family": "fixed_label", **t}


def fixed_extra(reader: str, feature: str) -> list[dict]:
    t = FIXED_NUMERIC.get(reader, {}).get(feature)
    return [{"id": f"{reader}:{feature}:fixed_numeric", "family": "fixed_numeric", **t}] if t else []


def fill(template: dict, value: str, word: str | None = None) -> str:
    """Candidate string for one feature value (``word`` for item-specific values)."""
    f = template["fillers"]
    slot = f["*"] if "*" in f else f[value]
    if word is not None:
        slot = slot.replace("{w}", word).replace("{W}", word[:1].upper() + word[1:])
    return template["template"].replace("{v}", slot)


def validate(template: dict, reader: str, feature: str) -> str | None:
    """Return an error string if a template breaks the native-format / differ-only-in-value rules."""
    t = template.get("template", "")
    fl = template.get("fillers", {})
    if not isinstance(t, str) or t.count("{v}") != 1:
        return "template must contain exactly one {v}"
    if reader == "ol" and not t.startswith("- "):
        return "OL template must start with '- '"
    vals = FEATURES[feature]["values"]
    if vals is None:
        if set(fl) != {"*"} or not any(k in fl["*"] for k in ("{w}", "{W}")):
            return "typo template needs fillers {'*': '...{w}...'}"
        return None
    if set(fl) != set(vals):
        return f"fillers must have keys {vals}"
    a, b = (fill(template, v) for v in vals)
    if a == b or not all(isinstance(fl[v], str) and fl[v].strip() for v in vals):
        return "filled candidates must differ and fillers be non-empty"
    if len(t) > 160:
        return "template too long"
    return None
