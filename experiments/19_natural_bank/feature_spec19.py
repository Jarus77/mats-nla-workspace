"""Exp 19 rows (notes/prereg_natural_bank.md §4): the same kind of concept as exp 18, but every item is a read point
inside REAL text from public sources (pip and npm package sources on this machine, arXiv e-prints), with both values
of a row drawn from the same file so the two classes share vocabulary. Items follow exp 13's shape:
{id, feature, cluster, value, text, meta, split, ids}; the read position is the last token of ``text``. The first
value is the marked one. ``both_positive`` marks rows whose two values are both positive states (ordinal / depth)
rather than a state and its absence; it is the antecedent of the revised complement rule."""
from __future__ import annotations

N_MINE_PER_VALUE = 20
N_SELECT_PER_VALUE = 80
N_PER_VALUE = N_MINE_PER_VALUE + N_SELECT_PER_VALUE

FEATURES: dict[str, dict] = {
    "py_string":  {"values": ["inside", "outside"], "next_tokens": ['"', "'"], "constrains_next": True, "short_value": True, "both_positive": False,
                   "desc": "Python source: read inside an open single-line string literal vs at a name a few tokens after its closing quote"},
    "js_string":  {"values": ["inside", "outside"], "next_tokens": ['"', "'"], "constrains_next": True, "short_value": True, "both_positive": False,
                   "desc": "JavaScript source: read inside an open string literal vs at a name after its closing quote"},
    "latex_math": {"values": ["inside", "outside"], "next_tokens": ["$"], "constrains_next": True, "short_value": True, "both_positive": False,
                   "desc": "arXiv LaTeX: read inside an open inline $...$ span vs in the prose after a closed one"},
    "py_arg":     {"values": ["third", "first"], "next_tokens": [",", ")"], "constrains_next": True, "short_value": True, "both_positive": True,
                   "desc": "Python call: read at the end of the 3rd positional argument vs the 1st, same call"},
    "json_depth": {"values": ["deep", "shallow"], "next_tokens": ["}"], "constrains_next": True, "short_value": True, "both_positive": True,
                   "desc": "JSON: read at the closing quote of a string value three or more objects deep vs at the top level"},
    "html_depth": {"values": ["deep", "shallow"], "next_tokens": ["</"], "constrains_next": True, "short_value": True, "both_positive": True,
                   "desc": "HTML page: read inside element text with six or more open elements vs at most three (html and body count)"},
    "md_heading": {"values": ["deep", "shallow"], "next_tokens": [], "constrains_next": False, "short_value": True, "both_positive": True,
                   "desc": "Markdown: read a few words into a level-3/4 heading vs a level-1/2 heading"},
    "md_list":    {"values": ["nested", "top"], "next_tokens": [], "constrains_next": False, "short_value": True, "both_positive": True,
                   "desc": "Markdown: read a few words into an indented (nested) list item vs a top-level one"},
}

FIXED: dict[str, dict[str, dict]] = {
    "nla": {
        "py_string":  {"template": "The current position is {v} a string literal", "fillers": {"inside": "inside", "outside": "outside"}},
        "js_string":  {"template": "The current position is {v} a string literal", "fillers": {"inside": "inside", "outside": "outside"}},
        "latex_math": {"template": "The current position is {v} a LaTeX math expression", "fillers": {"inside": "inside", "outside": "outside"}},
        "py_arg":     {"template": "The current token is the {v} argument of the function call", "fillers": {"third": "third", "first": "first"}},
        "json_depth": {"template": "The current value sits {v} in the JSON object", "fillers": {"deep": "three levels deep", "shallow": "at the top level"}},
        "html_depth": {"template": "The current text is nested {v} in the HTML", "fillers": {"deep": "three tags deep", "shallow": "one tag deep"}},
        "md_heading": {"template": "The current line is a {v} markdown heading", "fillers": {"deep": "lower-level", "shallow": "top-level"}},
        "md_list":    {"template": "The current line is a {v} list item", "fillers": {"nested": "nested", "top": "top-level"}},
    },
    "ol": {
        "py_string":  {"template": "- {v} string", "fillers": {"inside": "inside", "outside": "outside"}},
        "js_string":  {"template": "- {v} string", "fillers": {"inside": "inside", "outside": "outside"}},
        "latex_math": {"template": "- {v} math", "fillers": {"inside": "inside", "outside": "outside"}},
        "py_arg":     {"template": "- {v} argument", "fillers": {"third": "third", "first": "first"}},
        "json_depth": {"template": "- {v}", "fillers": {"deep": "nested three deep", "shallow": "top level"}},
        "html_depth": {"template": "- {v} nesting", "fillers": {"deep": "deep", "shallow": "shallow"}},
        "md_heading": {"template": "- {v} heading", "fillers": {"deep": "subsection", "shallow": "top"}},
        "md_list":    {"template": "- {v} list item", "fillers": {"nested": "nested", "top": "top-level"}},
    },
}

THRESH = {"readable": 0.80, "span": 0.80, "stmt": 0.30, "used_z": 3.0, "jenr": 1.5}
LEX_MAX = 0.65            # bank admission: bag-of-tokens AUROC on the last 48 tokens must stay below this
LAYERS_CAPTURE = (36, 42, 44, 52)
LAYER = 42
JREM_RANK = 1536
REL_DOSE = 0.11
K_NULL = 20
SEED_NULL = 19

CELL_THRESH = {"nla_prompted": 0.85, "nla_states": 0.20, "nla_names_complement": 0.20,
               "ol_prompted": 0.70, "ol_exhibits": 0.30, "ol_states": 0.05, "jlens_prompted": 0.70}
