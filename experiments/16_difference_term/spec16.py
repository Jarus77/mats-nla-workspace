"""Exp 16: Neel's difference-term experiment (23 Oct drop-in), run in raw activation space.

"Convert the OLens and NLA outputs into reconstructions ... the NLA will be much better ... what's
in this difference term? If you freeze the OLens output and change the bit of the NLA output
orthogonal to the OLens output to be from a different prompt, what happens to the NLA if you run
it on this new activation?"

Both readers' outputs are reconstructed with the NLA's own reconstructor (raw space; the Oracle
Lens reconstructor's output is whitened and its moments are gated). Per item i with activation h_i:
  R_ol_i   = NLA-AR(OLens bullets_i)        (what the OLens output carries, in raw space)
  R_nla_i  = NLA-AR(NLA explanation_i)      (exp 14)
  res_i    = h_i - R_ol_i                   (what OLens's output misses: the "difference term")
  resn_i   = R_nla_i - R_ol_i               (the extra reconstruction the NLA's text buys)
Partner j(i): an item of the OTHER class from a different cluster (seeded). Constructed vectors:
  swap_res   = R_ol_i + res_j     (OLens part frozen, difference term from another prompt)
  swap_ol    = R_ol_j + res_i     (difference term frozen, OLens part from another prompt)
  swap_resn  = R_ol_i + resn_j    (as swap_res with the NLA-reconstruction difference term)
  ol_only    = R_ol_i             (OLens part alone)
  plus references h_i, R_nla_i.
Readout: NLA and OLens@42 prompted log-odds (exp 13/07 frozen phrasings) on every vector.
Statistic: does the readout follow i's class or j's class (AUROC against each label)?
Probes: retrained linear probe on R_ol, res and resn (which component carries the concept).
Rows: the two NLA-leads rows and the two OLens-reads controls from exp 14. Prediction: for
language the readout follows the OLens part; for table column / parenthesis it follows the
difference term.
"""
from __future__ import annotations

ROWS: dict[str, dict] = {
    "tb_tablecol": {"src": "13", "values": ["first", "late"], "role": "nla_leads"},
    "tb_paren": {"src": "13", "values": ["inside", "outside"], "role": "nla_leads"},
    "ij2_lang": {"src": "07", "values": ["French", "German"], "role": "control"},
    "bk_listidx": {"src": "07", "values": ["early", "late"], "role": "control"},
}
LAYER = 42
OL_GEN_LAYER = "44"          # exp 14 fresh OLens readouts at the trained layer nearest 42
VARIANTS = ("h", "r_nla", "r_ol", "swap_res", "swap_ol", "swap_resn", "ol_only")
SEED = 16
