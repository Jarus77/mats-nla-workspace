"""Exp 14: is the Oracle Lens reconstructor (the "ruler" OLens was trained against) blind to the
concepts OLens misses? Neel, 23 Oct call: "If you show that the AR is just terrible at dealing with
that concept, that would be pretty interesting. You could also just directly study what is in the
error of the AR when it tries reconstructing things."

Rows: the two length-controlled cells where the NLA leads OLens (table column, parenthesis) and two
positive controls OLens reads well (language, bullet index). All 200 items per row (mining +
selection), activations already captured at 40/42/44.
"""
from __future__ import annotations

ROWS: dict[str, dict] = {
    "tb_tablecol": {"src": "13", "values": ["first", "late"], "role": "nla_leads"},
    "tb_paren": {"src": "13", "values": ["inside", "outside"], "role": "nla_leads"},
    "ij2_lang": {"src": "07", "values": ["French", "German"], "role": "control"},
    "bk_listidx": {"src": "07", "values": ["early", "late"], "role": "control"},
}
OL_AR_LAYERS = (44, 40)       # the AR has tags for 20..60 step 4 and 63; no 42 tag
NLA_LAYER = 42
K_GENS = 2                    # fresh readouts per item per reader (all 200 items)
SPAN_TOKENS = 32              # AR training crops are 1..32 tokens
LONG_SPAN_TOKENS = 128        # secondary "long span" input
MAX_TEXT_TOKENS = 256         # cap for reader texts fed to the AR
AR_REPO, AR_SUBDIR = "agu18dec/olens_and_ar", "ar_ptag_pooled"
RECON_KINDS = ("span32", "span128", "ol_bullets", "nla_expl")
