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

## Results (27 Oct, 18:35 UTC; `results/table24.md`, `results/recon24_{row}.json`; 160 selection items per row; one H100, 14 minutes, about $1)

Frozen-probe AUROC through the OL-AR at layer 44 (probe fit on the true-span rebuilds, applied frozen to each
condition's rebuild), with the grouped cross-validated probe on the condition alone in parentheses:

| row | span_32 | bag_32 | Oracle Lens readout | same, shuffled | NLA explanation | same, shuffled |
|---|---:|---:|---:|---:|---:|---:|
| py_string | 1.00 | 0.55 (0.50) | 0.80 (0.97) | 0.61 (0.73) | 0.56 (0.89) | 0.53 (0.70) |
| js_string | 0.97 | 0.44 (0.63) | 0.80 (0.92) | 0.54 (0.60) | 0.51 (0.81) | 0.42 (0.65) |
| latex_math | 0.99 | 0.59 (0.67) | 0.78 (0.93) | 0.58 (0.68) | 0.44 (0.81) | 0.50 (0.72) |
| py_arg | 0.97 | 0.53 (0.47) | 0.61 (0.58) | 0.46 (0.36) | 0.52 (0.59) | 0.50 (0.55) |
| json_depth | 0.98 | 0.71 (0.85) | 0.74 (0.92) | 0.67 (0.76) | 0.51 (0.83) | 0.53 (0.72) |
| html_depth | 0.99 | 0.74 (0.98) | 0.73 (0.78) | 0.62 (0.76) | 0.44 (0.90) | 0.57 (0.66) |
| md_heading | 0.63 | 0.51 (0.50) | 0.47 (0.57) | 0.55 (0.52) | 0.44 (0.60) | 0.46 (0.50) |
| md_list | 0.59 | 0.53 (0.60) | 0.50 (0.68) | 0.57 (0.55) | 0.46 (0.49) | 0.47 (0.58) |
| tb_tablecol | 1.00 | 0.62 (0.57) | 0.57 (0.71) | 0.45 (0.64) | 0.47 (0.86) | 0.48 (0.50) |
| tb_paren | 0.94 | 0.64 (0.63) | 0.60 (0.77) | 0.51 (0.67) | 0.69 (0.77) | 0.53 (0.64) |

Cosine of each condition's class-mean direction with the true span's (layer 44): Oracle Lens readout 0.31 to 0.58 on
the six readable natural rows, 0.37 / 0.32 on the position rows; shuffled 0.00 to 0.26; NLA explanation -0.10 to
0.16 everywhere; bag_32 0.01 to 0.32. Layer 40 repeats every number within 0.04 (`table24.md`, lower block).

### Scoring the predictions

| prediction | result |
|---|---|
| 1. both readers' intact readouts >= 0.70 frozen on the six readable rows | Oracle Lens 5 of 6 (py_arg 0.61); NLA 0 of 6 (0.44 to 0.56). By the cross-validated probe instead: Oracle Lens 5 of 6, NLA 5 of 6 (py_arg 0.58 / 0.59 the miss both times) |
| 2. shuffling hurts the position rows, not the string rows | wrong about the string rows: shuffling costs 0.19 to 0.26 frozen (0.24 to 0.32 cross-validated) on py_string and js_string too. On the position rows Oracle Lens's intact readout is already at the bag level frozen (0.57 / 0.60 against 0.62 / 0.64) and modestly above it cross-validated (0.71 / 0.77 against 0.57 / 0.63) |
| 3. Markdown rows near 0.5 | holds (0.44 to 0.68) |

## Reading

1. **Oracle Lens's readouts carry string, math and depth state through a decoder with no stake in either reader.**
   Frozen 0.73 to 0.80 on five of six readable rows, cross-validated 0.78 to 0.97, class direction 0.31 to 0.58 of the
   way to the true span's. Lower than through the NLA critic (exp 20: 0.70 to 0.94) with the same ordering, and the
   one row under the line is the same (call argument: 0.70 there, 0.61 here). The exp 20 reading "Oracle Lens's
   readout re-induces the state, without naming it" does not depend on the critic.
2. **The NLA's explanations carry the state, but not along the span's direction.** Frozen AUROC is at chance and the
   class-direction cosine is zero, yet a probe fit on the explanations' own rebuilds separates the classes at 0.81
   to 0.90 on five of six readable rows and at 0.86 on table column. The OL-AR was trained on spans; it maps a
   description to a region unlike the span's image, and inside that region the two classes sit apart. This is the
   in-distribution part of exp 20's NLA column made visible: through its own critic the explanation lands on the
   span direction (cosine 0.77 to 0.90), through a span-trained decoder it does not. So two claims separate. The
   weak one, "the state is recoverable from the NLA's explanation", holds under both decoders and the text bag. The
   strong one, "the explanation re-induces the state as the span would", is a property of the NLA's critic. The
   thesis sentence should say that the readers' outputs carry the state, Oracle Lens's the way a continuation does,
   the NLA's the way a description does.
3. **Under this decoder the carrying is arrangement, not vocabulary, for both readers.** Shuffling a readout's tokens
   costs 0.19 to 0.32 AUROC on every row where the intact readout carried the state, string rows included. The OL-AR
   is a window map (bag_32 0.44 to 0.74 against span_32 0.97 to 1.00), so a decoder that reads text as the model
   reads it needs the order. Exp 20's bag-of-tokens classifier reached 0.96 / 0.88 on the string rows because a bag
   classifier is free to learn vocabulary cues; a reading decoder is not. Prediction 2 was wrong about the decoder,
   not about the readouts: the state is in the words and in their arrangement, and which one a decoder uses is the
   decoder's property.
4. **Position rows.** Oracle Lens's readout carries table column and parenthesis at the bag level by the frozen
   measure and modestly above it cross-validated, in line with exp 16 (0.01 of the position signal in the part of the
   vector Oracle Lens's words rebuild, same decoder, layer 44) and exp 20 (0.64 / 0.67 through the critic). The two
   decoders agree: Oracle Lens's words carry little position; the NLA's description carries table column (0.86
   cross-validated) by describing the format, as exp 20 found.
5. **What does not change.** Neither readout names any of these states (exp 20 examples, exp 23); the Markdown rows
   are at chance through every decoder, consistent with their probe AUROC of 0.63 / 0.65 at layer 42.

## What this changes

The "both readers carry it" sentence survives a second decoder, with one wording change for the NLA (recoverable
from the explanation, not re-induced along the span direction). The exp 20 NLA column (0.93 to 0.99) should be
quoted together with this one and the text bag, not alone. The vocabulary-versus-arrangement question is answered
"both, and the decoder decides"; the derivation note's consequence C5 is updated to say so.
