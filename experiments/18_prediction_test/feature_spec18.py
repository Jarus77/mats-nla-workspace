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
