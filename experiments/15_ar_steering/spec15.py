"""Exp 15: does the model use the position-in-text state the NLA reports, and can the reconstructor
build a steering vector for it from text?

Neel (23 Oct drop-in, to the OLens red-team group): get "causal evidence ... especially using the
activation reconstructor": map reader outputs to vectors, test presence, ablate. To Kylie: "take an
AR from Oracle Lens and see if it's already solved [steering-vector generation]".

Design (one concept at a time; both exp 13 rows where the NLA leads OLens, length-controlled):
  directions at L42, all unit, "+" points toward values[1]:
    cm_raw / cm_jrem4        class-mean (values[1] - values[0]) of the 200 natural activations,
                             raw and with the top-4r* J-space projected out (exp 08 recipe)
    arexpl_raw / arexpl_jrem4  class-mean of the NLA-AR reconstructions of the NLA's OWN explanations
                             (exp 14 recon14_nla: 200 items x 2 readouts): "AR from exhibited text"
    arspan_raw / arspan_jrem4  class-mean of NLA-AR reconstructions of the last 32 real tokens
    arstmt_raw / arstmt_jrem4  NLA-AR(fixed statement B) - NLA-AR(fixed statement A): "AR from a statement"
  construction: centred, h' = h - (v.h) v + s * dose * v (exp 08 primary), s in {+1, -1}
  doses: r11 = 0.11 ||h_i|| per item (one norm for every direction and null); q90 per real direction
  nulls: K = 20 isotropic + K = 20 J-complement (4r*) unit directions at the r11 dose
  model side (base Qwen, edit at the read token only, prefill): next-token logits + 32-token greedy
    continuation -> tb_tablecol: cells (| separators) emitted before the first newline;
                    tb_paren: tokens until the first ')' and next-token mass on ')' tokens
  reader side: NLA, OLens@42, J-lens@42 prompted log-odds on the steered vectors (exp 13 frozen
    phrasings: fixed label + selected mined templates; J-lens word pairs from feature_spec)
  statistic: per-item paired shift (+ minus -), real cell vs the K null contrasts (z, band)
"""
from __future__ import annotations

ROWS: dict[str, dict] = {
    "tb_tablecol": {"values": ["first", "late"], "model_metric": "cells_to_newline", "expect_sign": -1},
    "tb_paren": {"values": ["inside", "outside"], "model_metric": "tokens_to_close", "expect_sign": +1},
}
LAYER = 42
RANKS = (384, 1536)          # r*, 4r* (prereg)
JREM_RANK = 1536
DIR_NAMES = ("cm_raw", "cm_jrem4", "arexpl_raw", "arexpl_jrem4", "arspan_raw", "arspan_jrem4",
             "arstmt_raw", "arstmt_jrem4")
REL_DOSE = 0.11              # r11: 0.11 * ||h_i||
K_NULL = 20
NULL_CLASSES = ("iso", "nonj4")
GEN_TOKENS = 32
SPAN_TOKENS = 32
JLENS_WORDS = {"tb_tablecol": {"first": ["first", "one"], "late": ["fifth", "five"]},
               "tb_paren": {"inside": ["inside", "parenthesis"], "outside": ["outside"]}}
