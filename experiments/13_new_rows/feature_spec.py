"""Exp 13: new concept rows for the per-reader recall table (Neel's thread, 11 Oct).

Five position-in-text rows (the kind Neel called interesting), two formatting rows (Camila's kind, for
comparison), and a fresh line-position bank scored with the frozen exp 07 col templates. Same item
shape and splits as exp 07: {id, feature, split, cluster, value, ids, text, meta}; 20 + 20 mining
items and 80 + 80 selection items per feature.
"""
from __future__ import annotations

N_MINE_PER_VALUE = 20
N_SELECT_PER_VALUE = 80
K_READOUTS = 8
OL_LAYERS = (42, 40, 44)
NLA_LAYER = 42
N_SELECT_TEMPLATES = 3

BINS: dict[str, dict[str, tuple[int, int]]] = {
    "tb_sentcount": {"early": (1, 2), "late": (5, 9)},       # 1-based index of the current sentence
    "tb_tablecol": {"first": (1, 1), "late": (4, 6)},        # 1-based column of the current cell
    "tb_codeline": {"early": (2, 3), "late": (9, 12)},       # 1-based body line of the function
    "tb_wordsleft": {"many": (20, 25), "few": (2, 5)},       # words remaining under the stated limit
    "fmt_listdepth": {"shallow": (1, 1), "deep": (3, 3)},    # markdown nesting depth
}
SENT_WORDS_IN = (6, 15)        # words into the current sentence, matched across sentcount values
TABLE_ROWS = (3, 6)            # data-row index of the read cell, matched across tablecol values
WORDSLEFT_LIMIT = (30, 60)     # the stated word limit N
QA_WORDS_IN = (15, 30)         # words into the answer / passage at the read token

FEATURES: dict[str, dict] = {
    "tb_sentcount": {
        "values": ["early", "late"],
        "desc": "how many sentences into the current paragraph the read token is, in English prose: "
                "'early' = inside the 1st or 2nd sentence, 'late' = inside the 5th sentence or later; "
                "read 6-15 words into the sentence in both cases",
    },
    "tb_tablecol": {
        "values": ["first", "late"],
        "desc": "which column of a markdown table the read token is in: 'first' = the first cell of "
                "the row, 'late' = the 4th to 6th cell; read inside a cell of a data row",
    },
    "tb_codeline": {
        "values": ["early", "late"],
        "desc": "how many lines into a flat Python function body the read token is: 'early' = body "
                "line 2-3, 'late' = body line 9-12; all lines at the same indentation",
    },
    "tb_paren": {
        "values": ["inside", "outside"],
        "desc": "whether the read token is inside an open parenthesis in English prose: 'inside' = 3+ "
                "words into a parenthetical that has not closed, 'outside' = 4+ words after a closing "
                "parenthesis and not inside any",
    },
    "tb_wordsleft": {
        "values": ["many", "few"],
        "desc": "how many words remain under an explicit instruction 'Write exactly N words': 'many' = "
                "20-25 words still to go, 'few' = 2-5 words still to go; read at the end of a word",
    },
    "fmt_qa": {
        "values": ["qa", "prose"],
        "desc": "whether the text is laid out as a question-and-answer exchange ('Question: ... "
                "Answer: ...') or as plain prose with the same content; read 15-30 words into the answer",
    },
    "fmt_listdepth": {
        "values": ["shallow", "deep"],
        "desc": "nesting depth of the current item in an indented markdown bullet list: 'shallow' = "
                "top level, 'deep' = three levels in; read inside the item text",
    },
    "nj1_col_fresh": {
        "values": ["short", "long"],
        "desc": "characters since the last newline (current column) in hard-wrapped prose: 'short' = "
                "10-30 characters into the line, 'long' = 55-85; fresh passages, scored with the "
                "frozen exp 07 nj1_col templates only (no mining)",
        "select_only": True,
    },
}

# Fixed label templates (never mined), one per reader, as in exp 07.
FIXED: dict[str, dict[str, dict]] = {
    "nla": {
        "tb_sentcount": {"template": "This is the {v} sentence of the paragraph",
                         "fillers": {"early": "second", "late": "sixth"}},
        "tb_tablecol": {"template": "The current cell is in the {v} column of the table",
                        "fillers": {"first": "first", "late": "fifth"}},
        "tb_codeline": {"template": "This is line {v} of the function body",
                        "fillers": {"early": "two", "late": "ten"}},
        "tb_paren": {"template": "The current position is {v} a parenthesis",
                     "fillers": {"inside": "inside", "outside": "outside"}},
        "tb_wordsleft": {"template": "About {v} words remain under the word limit",
                         "fillers": {"many": "twenty", "few": "three"}},
        "fmt_qa": {"template": "The text is formatted as {v}",
                   "fillers": {"qa": "a question and answer", "prose": "plain prose"}},
        "fmt_listdepth": {"template": "The current list item is nested {v}",
                          "fillers": {"shallow": "at the top level", "deep": "three levels deep"}},
    },
    "ol": {
        "tb_sentcount": {"template": "- sentence {v}", "fillers": {"early": "two", "late": "six"}},
        "tb_tablecol": {"template": "- column {v}", "fillers": {"first": "one", "late": "five"}},
        "tb_codeline": {"template": "- line {v}", "fillers": {"early": "two", "late": "ten"}},
        "tb_paren": {"template": "- {v} parentheses", "fillers": {"inside": "inside", "outside": "outside"}},
        "tb_wordsleft": {"template": "- {v} words left", "fillers": {"many": "twenty", "few": "three"}},
        "fmt_qa": {"template": "- {v} format", "fillers": {"qa": "Q&A", "prose": "prose"}},
        "fmt_listdepth": {"template": "- nesting level {v}", "fillers": {"shallow": "one", "deep": "three"}},
    },
}


# ---------------------------------------------------------------- exp 07 template API (same names)
import re  # noqa: E402

QUALITATIVE_PRIMARY: tuple[str, ...] = ()
FIXED_NUMERIC: dict[str, dict] = {}
BREAK_SOON_MAX, BREAK_FAR_MIN, BREAK_COL_RANGE = 6, 25, (40, 75)  # unused here; select_freeze imports them

FIXED_JLENS: dict[str, dict[str, list[str]]] = {
    "tb_sentcount": {"early": ["second", "two"], "late": ["sixth", "six"]},
    "tb_tablecol": {"first": ["first", "one"], "late": ["fifth", "five"]},
    "tb_codeline": {"early": ["two", "second"], "late": ["ten", "tenth"]},
    "tb_paren": {"inside": ["inside", "parenthesis"], "outside": ["outside"]},
    "tb_wordsleft": {"many": ["twenty", "many"], "few": ["three", "few"]},
    "fmt_qa": {"qa": ["question", "answer"], "prose": ["prose", "paragraph"]},
    "fmt_listdepth": {"shallow": ["top", "one"], "deep": ["three", "nested"]},
    "nj1_col_fresh": {"short": ["twenty"], "long": ["seventy"]},
}

_NUMWORD = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                      r"\w+teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
                      re.IGNORECASE)


def is_numeric(template: dict) -> bool:
    return any(_NUMWORD.search(template["template"].replace("{v}", f)) for f in template["fillers"].values())


def fixed_template(reader: str, feature: str) -> dict:
    t = FIXED[reader][feature]
    return {"id": f"{reader}:{feature}:fixed", "family": "fixed_label", **t}


def fixed_extra(reader: str, feature: str) -> list[dict]:
    t = FIXED_NUMERIC.get(reader, {}).get(feature)
    return [{"id": f"{reader}:{feature}:fixed_numeric", "family": "fixed_numeric", **t}] if t else []


def fill(template: dict, value: str, word: str | None = None) -> str:
    f = template["fillers"]
    slot = f["*"] if "*" in f else f[value]
    if word is not None:
        slot = slot.replace("{w}", word).replace("{W}", word[:1].upper() + word[1:])
    return template["template"].replace("{v}", slot)


def validate(template: dict, reader: str, feature: str) -> str | None:
    t = template.get("template", "")
    fl = template.get("fillers", {})
    if not isinstance(t, str) or t.count("{v}") != 1:
        return "template must contain exactly one {v}"
    if reader == "ol" and not t.startswith("- "):
        return "OL template must start with '- '"
    vals = FEATURES[feature]["values"]
    if set(fl) != set(vals):
        return f"fillers must have keys {vals}"
    a, b = (fill(template, v) for v in vals)
    if a == b or not all(isinstance(fl[v], str) and fl[v].strip() for v in vals):
        return "filled candidates must differ and fillers be non-empty"
    if len(t) > 160:
        return "template too long"
    return None
