"""Exp 18 rows (notes/prereg_prediction_test.md §4). Items follow exp 13's shape: {id, feature, cluster,
value, text, meta, split, ids}; the read position is the last token of ``text``. Two values per row; the
first value is the "marked" one where the row has one. ``next_tokens`` is the declared model-side token
set for the USED instrument ("none" = scored not-used by construction). Flags are the §3 antecedents."""
from __future__ import annotations

N_MINE_PER_VALUE = 20
N_SELECT_PER_VALUE = 80
N_GEN = 2 * (N_MINE_PER_VALUE + N_SELECT_PER_VALUE) + 20   # per synthetic row, before split assignment

FEATURES: dict[str, dict] = {
    "json_depth":  {"values": ["deep", "shallow"], "next_tokens": ["}"], "constrains_next": True, "short_value": True,
                    "desc": "JSON: read at the end of a string value nested three objects deep vs at the top level"},
    "latex_math":  {"values": ["inside", "outside"], "next_tokens": ["$"], "constrains_next": True, "short_value": True,
                    "desc": "LaTeX prose: read 3+ tokens into an open $...$ math span vs 4+ words after a closed one"},
    "py_string":   {"values": ["inside", "outside"], "next_tokens": ['"', "'"], "constrains_next": True, "short_value": True,
                    "desc": "Python: read inside an open string literal vs after its closing quote, same statement shape"},
    "call_arg":    {"values": ["third", "first"], "next_tokens": [",", ")"], "constrains_next": True, "short_value": True,
                    "desc": "Python call: read at the end of the 3rd positional argument vs the 1st"},
    "md_heading":  {"values": ["h3", "h1"], "next_tokens": [], "constrains_next": False, "short_value": True,
                    "desc": "Markdown: read 4-8 words into a level-3 heading vs a level-1 heading"},
    "csv_col":     {"values": ["fourth", "first"], "next_tokens": [",", "\n"], "constrains_next": True, "short_value": True,
                    "desc": "CSV data row: read inside the 4th field vs the 1st field"},
    "html_depth":  {"values": ["deep", "shallow"], "next_tokens": ["</"], "constrains_next": True, "short_value": True,
                    "desc": "HTML: read inside text with three open tags vs one open tag"},
    "stance_word": {"values": ["unfortunately", "fortunately"], "next_tokens": [], "constrains_next": False, "short_value": True,
                    "desc": "prose paragraph opened with 'Unfortunately,' vs 'Fortunately,', read 15-30 words later (sayable-but-unused candidate)"},
}

# Fixed-label phrasings per reader (frozen here, before any reader runs). NLA: a sentence in its explanation
# register; OL: a bullet. {v} takes the filler for the value. Also the statement pair for the
# sayable-by-statement instrument (NLA sentence, both values).
FIXED: dict[str, dict[str, dict]] = {
    "nla": {
        "json_depth":  {"template": "The current value sits {v} in the JSON object", "fillers": {"deep": "three levels deep", "shallow": "at the top level"}},
        "latex_math":  {"template": "The current position is {v} a LaTeX math expression", "fillers": {"inside": "inside", "outside": "outside"}},
        "py_string":   {"template": "The current position is {v} a string literal", "fillers": {"inside": "inside", "outside": "outside"}},
        "call_arg":    {"template": "The current token is the {v} argument of the function call", "fillers": {"third": "third", "first": "first"}},
        "md_heading":  {"template": "The current line is a {v} markdown heading", "fillers": {"h3": "level-three", "h1": "level-one"}},
        "csv_col":     {"template": "The current token is in the {v} field of the CSV row", "fillers": {"fourth": "fourth", "first": "first"}},
        "html_depth":  {"template": "The current text is nested {v} in the HTML", "fillers": {"deep": "three tags deep", "shallow": "one tag deep"}},
        "stance_word": {"template": "The passage opened with the word {v}", "fillers": {"unfortunately": "unfortunately", "fortunately": "fortunately"}},
        "ij2_lang":    {"template": "The text is written in {v}", "fillers": {"French": "French", "German": "German"}},
        "bk_quote":    {"template": "The current position is {v} a quotation", "fillers": {"inside": "inside", "outside": "outside"}},
    },
    "ol": {
        "json_depth":  {"template": "- {v}", "fillers": {"deep": "nested three deep", "shallow": "top level"}},
        "latex_math":  {"template": "- {v} math", "fillers": {"inside": "inside", "outside": "outside"}},
        "py_string":   {"template": "- {v} string", "fillers": {"inside": "inside", "outside": "outside"}},
        "call_arg":    {"template": "- {v} argument", "fillers": {"third": "third", "first": "first"}},
        "md_heading":  {"template": "- {v} heading", "fillers": {"h3": "level three", "h1": "level one"}},
        "csv_col":     {"template": "- {v} field", "fillers": {"fourth": "fourth", "first": "first"}},
        "html_depth":  {"template": "- {v} nesting", "fillers": {"deep": "deep", "shallow": "shallow"}},
        "stance_word": {"template": "- {v}", "fillers": {"unfortunately": "unfortunately", "fortunately": "fortunately"}},
        "ij2_lang":    {"template": "- {v} language", "fillers": {"French": "French", "German": "German"}},
        "bk_quote":    {"template": "- {v} quotation", "fillers": {"inside": "inside", "outside": "outside"}},
    },
}

# Instrument thresholds and prediction rules (prereg §2-§3), as constants so the code and the document agree.
THRESH = {"readable": 0.80, "span": 0.50, "stmt": 0.30, "used_z": 3.0, "jenr": 1.5}
LAYERS_CAPTURE = (36, 42, 44, 52)
LAYER = 42
JREM_RANK = 1536
REL_DOSE = 0.11
K_NULL = 20
SEED_NULL = 18
