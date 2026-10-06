"""09 OL-mechanism pilot: FIXED channels and readouts (written BEFORE any 09 scoring; mtime recorded in
RESULTS.md). DECLARED PILOT SPLIT -- EXCLUDED FROM THE MAIN ANALYSIS.

Every OL "channel" is a list of (a, b) candidate pairs scored teacher-forced as OL's first '- ' bullet
(prefix ''), LO = log p(a | act) - log p(b | act), averaged over the pairs. Every model "readout" is the
matching next-token contrast on the model's own output distribution at the position after the read token
(mean over pairs of logsumexp(lp[A]) - logsumexp(lp[B])).

Re-instatement analogue: OL's slot after '-' plays the role of the model's next token. So
  OL "-" + t_a  vs  "-" + t_b      <->   model  log P(t_a) - log P(t_b)   (same tokens).

Random-direction channels (randA / randB) are filled in by the FIXED RULE below during the MODEL run
(prep stage), written to /results/09_ol_mechanism/random_selection.json BEFORE any OL scoring.
"""
from __future__ import annotations

SP4, SP8 = " " * 4, " " * 8
IND_WORDS = ("return", "if", "self")

# 07 frozen (templates_frozen.json sha256 09156722..., verbatim)
F07_TENSE = [("- They were", "- They are"), ("- The village had", "- The village has"), ("- It was", "- It is")]
F07_INDENT = [("- 2  # Multiple increments in nested scope", "- 1  # Header row counts as a pass"),
              ("- 3  # nested scope  # check indentation level", "- 1  # first level  # check indentation level"),
              ("- , count += 2  # multiple increments in same block", "- += 1; lines_count += 1")]  # deep, shallow

OL_CHANNELS = {
    # col (r5, fixed 00:53 in 08): newline-then-word vs word
    "nl_cont": [("- \nthe", "- the"), ("- \nand", "- and"), ("- \nof", "- of")],
    # tense token channel (new; + = past)
    "tense_tok": [("- was", "- is"), ("- were", "- are"), ("- had", "- has")],
    "f07_tense": F07_TENSE,
    # indentation (+ = deeper). ind_cont: indentation token right after '-' (tokens '-', 7sp / 3sp, ' w')
    "ind_cont": [("-" + SP8 + w, "-" + SP4 + w) for w in IND_WORDS],
    # indented continuation after a newline: '-', ' \n', 7sp / 3sp, ' w'
    "ind_nlcont": [("- \n" + SP8 + w, "- \n" + SP4 + w) for w in IND_WORDS],
    # indented vs flush continuation after a newline
    "ind_any": [("- \n" + SP4 + w, "- \n" + w) for w in IND_WORDS],
    "f07_indent": F07_INDENT,
}

# Which OL channels are scored on which item set (all conditions of the set, adapter on AND off).
SET_CHANNELS = {
    "codeNL": ("ind_cont", "ind_nlcont", "ind_any", "nl_cont", "f07_indent"),
    "codeIN": ("ind_cont", "ind_nlcont", "ind_any", "nl_cont", "f07_indent"),
    "col": ("nl_cont", "tense_tok", "randA", "randB", "randC", "randD"),
    "tense": ("tense_tok", "f07_tense", "nl_cont", "randA", "randB", "randC", "randD"),
}

# Model readouts (token strings; resolved to ids in the container). NL = whitespace-only newline tokens
# (pilot_lib.eol_token_ids 'narrow'); NLB = every newline-containing token (r5's behavioural set).
READOUTS = {
    "nl_vs_content": [("NL", [" the"]), ("NL", [" and"]), ("NL", [" of"])],  # analogue of nl_cont
    "nl_logit_broad": "NLB",                                                  # r5 behavioural readout
    "tense_tok": [([" was"], [" is"]), ([" were"], [" are"]), ([" had"], [" has"])],
    "tense_broad": [([" was", " were", " had", " did", " said", " became", " went", " made", " took", " came"],
                     [" is", " are", " has", " does", " says", " becomes", " goes", " makes", " takes", " comes"])],
    "ind8v4": [([" " * 7], [" " * 3])],  # next token = indentation of the next line (8 vs 4 spaces)
}
# For codeIN the model readout is taken one token later, after the true next token "\n" is appended
# (the read token is the last content token of its line).

# Primary (channel, readout) analogue per direction, used for the gain ratio.
ANALOGUE = {
    "col": ("nl_cont", "nl_vs_content"),
    "tense": ("tense_tok", "tense_tok"),
    "randA": ("randA", "randA"),
    "randB": ("randB", "randB"),
    "randC": ("randC", "randC"),
    "randD": ("randD", "randD"),
    "indent": ("ind_cont", "ind8v4"),     # codeNL; codeIN uses ("ind_nlcont", "ind8v4")
    "nextind": ("ind_cont", "ind8v4"),
}

# ---------------------------------------------------------------- random-direction FIXED RULE
RANDOM_RULE = {
    "n_candidates": 32,
    "kind": "nonj (J-complement at 4r* = 1536, L42 basis)",
    "seed": "909000 + k",
    "dose": "centred, 0.108 * ||h_i||, +/-",
    "selection_items": "sel09 (60 fresh prose items, disjoint from all eval items)",
    "eligible_tokens": "decoded matches ^ [a-z]{3,}$, natural mean P >= 1e-4 on sel09 (zero dose), "
                       "not in {the, and, of, was, is, were, are, had, has}",
    "per_token_effect": "D_t = mean_i (log p_t(+) - log p_t(-)) / 2 at the next position",
    "readout": "a_1..3 = top-3 D_t (most increased), b_1..3 = bottom-3 (most decreased); pairs (a_k, b_k)",
    "score": "mean_k D_{a_k} - D_{b_k}",
    "choose": "the 2 candidates with the largest score -> randA (largest), randB (second)",
    "ol_channel": "pairs ('-' + a_k, '-' + b_k) (decoded tokens keep their leading space)",
}
# AMENDMENT A1 (2026-10-06 ~06:50 IST; after the MODEL-ONLY selection run, BEFORE any OL scoring): the nonj
# pool's chosen readouts moved only ~0.09 nats on sel09 (col 4r* moves newline ~0.7), i.e. not "comparable".
# Keep randA/randB and add an ISOTROPIC pool under the identical rule -> randC (largest), randD (second);
# random directions get doses r108 and r216 (r216 for signal-to-noise; the ratio is per unit dose).
RANDOM_RULE_A1 = {**RANDOM_RULE, "kind": "iso (isotropic)", "seed": "919000 + k",
                  "choose": "the 2 candidates with the largest score -> randC (largest), randD (second)"}
EXCLUDE_RAND_TOKENS = {" the", " and", " of", " was", " is", " were", " are", " had", " has"}
