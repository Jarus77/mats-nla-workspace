"""Exp 17: harden the headline cells (table column, parenthesis) before they reach Neel.

(a) Multi-candidate re-score: Guiv's critique is that a two-candidate log-odds oversells a reader
    (85-94% two-way fell to 12-21% with eight candidates on his bank). Here every value-slot phrasing
    is scored over the full value set and the metrics are top-1 among C, MRR, and the two-class AUROC
    re-normalised over all C candidates (log-sum-exp of each class's candidates).
(b) Cluster-bootstrap CIs for the exp 15 r11 shifts and the exp 16 follow-AUROCs (ci_prior.py).
(c) Vocabulary readout of the exp 15 directions through the unembedding (is AR(statement) lexical?).
(d) Cross-scoring with Guiv's bank once his files are in inputs/ (skip otherwise).
Declared criteria: docs/superpowers/specs/2026-10-26-harden-headline-design.md.
"""
from __future__ import annotations

ROWS: dict[str, dict] = {
    "tb_tablecol": {"values": ["first", "late"], "n_cands": 6},
    "tb_paren": {"values": ["inside", "outside"], "n_cands": 4},
}
COL_ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth"]
COL_CARDINAL = ["one", "two", "three", "four", "five", "six"]
COL_DIGIT = ["1", "2", "3", "4", "5", "6"]
PAREN_STATES = ["paren_in", "paren_out", "quote_in", "bracket_in"]

# Value-slot phrasings only (the binary mined NLA families such as "Final token begins a new row" have no
# slot for a third value and stay two-candidate; they are reported from exp 13 as the reference).
MC_TEMPLATES: dict[str, dict[str, list[dict]]] = {
    "tb_tablecol": {
        "nla": [{"id": "nla:mc:fixed", "template": "The current cell is in the {v} column of the table",
                 "slots": COL_ORDINAL}],
        "ol": [{"id": "ol:mc:fixed", "template": "- column {v}", "slots": COL_CARDINAL},
               {"id": "ol:mc:m3", "template": "- col_{v} |", "slots": COL_DIGIT},
               {"id": "ol:mc:m0", "template": "- column {v} |", "slots": COL_DIGIT}],
        "jlens": [{"id": "jlens:mc:fixed", "template": None,
                   "slots": [["first", "one"], ["second", "two"], ["third", "three"],
                             ["fourth", "four"], ["fifth", "five"], ["sixth", "six"]]}],
    },
    "tb_paren": {
        "nla": [{"id": "nla:mc:state", "template": "The current position is {v}",
                 "slots": ["inside a parenthesis", "outside any parenthesis", "inside a quotation",
                           "inside square brackets"]}],
        "ol": [{"id": "ol:mc:state", "template": "- {v}",
                "slots": ["inside parentheses", "outside parentheses", "inside quotes", "inside brackets"]}],
        "jlens": [{"id": "jlens:mc:state", "template": None,
                   "slots": [["inside", "parenthesis"], ["outside"], ["quotation", "quote"],
                             ["bracket", "brackets"]]}],
    },
}
# 0-based candidate indices that belong to each class value.
CLASS_SETS: dict[str, dict[str, list[int]]] = {
    "tb_tablecol": {"first": [0], "late": [3, 4, 5]},
    "tb_paren": {"inside": [0], "outside": [1]},
}
VALID_COLS = {1, 4, 5, 6}
N_BOOT = 2000
SEED = 17
LAYER = 42
OL_LAYERS = (42,)
