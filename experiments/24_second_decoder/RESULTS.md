# Exp 24: a second decoder for what the readers' outputs carry

## Question

Exp 20 found that both readers' free readouts, fed through the NLA's own reconstructor (the critic), rebuild the
structural state of the read point (inside a string, call-argument index, JSON/HTML depth, table column, inside a
parenthesis) at 0.70 to 0.99 frozen-probe AUROC on every readable row, although neither readout ever names that state.
Does the same hold through a decoder the NLA was never trained against, and is what the readouts carry their
vocabulary or their arrangement?

## Why a second decoder

The exp 20 critic was trained on NLA explanations, so the NLA column there is in-distribution (its RESULTS.md, reading
5) and the "carrying" claim for both readers rests on one decoder with a stake in one of them. Exp 20's text-bag
cross-check removed the critic but with it any sensitivity to word order. Exp 14 established a second reconstructor:
the Oracle Lens activation reconstructor (OL-AR, `agu18dec/olens_and_ar:ar_ptag_pooled`), trained on text spans and
never on either reader's output, with tags at layers 20..60 step 4 and 63 (no 42; exp 14 used 44 as primary and 40 as
a check, and so does this). Its head output lives in a whitened space whose moments are gated, so, as in exp 14,
nothing here is compared with a raw activation: the reference for every row is the OL-AR's own rebuild of the true
last-32-token span, and every condition is scored inside that output space. If the OL-AR agrees with the critic on
which readouts carry which states, the finding does not depend on the NLA's decoder; the shuffled-readout control then
says whether the carrying is vocabulary (survives the shuffle) or arrangement (does not).

## Pipeline

One Modal stage (`modal24.py`, app `t6bn`, one H100; `run24.sh` chains launch, poll, fetch and `analyze24.py`):

- Rows: exp 20's eight natural rows (exp 19's bank) plus exp 13's `tb_tablecol` and `tb_paren`; selection split
  only, 160 items per row, labels = exp 20's MARKED value, grouped folds by item cluster.
- Inputs, all read from the volume: exp 20's readouts `ol_sel_{row}.json` and `nla_sel_{row}.json` (k = 1, layer 42)
  and the items' token ids. No activations are used.
- Conditions per item, each fed to the OL-AR as token ids after the layer tag (readouts encoded with the Qwen
  tokenizer, capped at exp 14's 256 tokens, empty ones become `(empty)`): `span_32` (true last 32 tokens, the
  reference), `span_128`, `bag_32` (the same 32 tokens in random order, seed 24), `ol_readout`, `ol_readout_shuffled`
  (the readout's tokens in random order, seed 24), `nla_readout`, `nla_readout_shuffled`.
- Decoder: `mats.readers.olens_ar.OLensAR` (lifted from exp 14), tags 44 and 40, head/norm variant fixed to exp 14's
  calibrated choice `li/ln` (`experiments/14_ar_blindspot/results/calib14.json`; recorded per row as `variant`).
- Scores per layer and condition, all inside the OL-AR output space: (a) frozen-probe AUROC: a probe fit on the
  `span_32` rebuilds (grouped 5-fold), applied frozen to the condition's rebuild of the held-out items; (b) grouped
  cross-validated probe AUROC on the condition's rebuild alone (`mats.probes.cv_auroc`); (c) cosine of the
  condition's class-mean direction with `span_32`'s.
- Outputs: `/results/24_second_decoder/recon24_{row}.json` (resumable: finished rows are skipped), tables in
  `results/table24.md` and `results/table24.json` (layer 44, layer 40 beneath).

## Predictions written before the run

1. Through the OL-AR at layer 44, both readers' intact readouts keep frozen-probe AUROC >= 0.70 on the six readable
   natural rows (py_string, js_string, latex_math, py_arg, json_depth, html_depth), as they did through the NLA
   critic in exp 20.
2. The shuffled readouts fall toward the `bag_32` level on the two position rows (tb_tablecol, tb_paren), where
   exp 20's text-bag check was weak (0.71 / 0.56 for Oracle Lens), and stay high on the string rows, where the text
   bag was already high (0.96 / 0.88): string state is carried by vocabulary, position state by arrangement.
3. The two Markdown rows (md_heading, md_list) stay near 0.5 everywhere.

## Results

(pending)
