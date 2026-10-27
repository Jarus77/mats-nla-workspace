# Exp 25: the reconstructor-based "exhibits" measure on the exp 22 public bank

**Question.** When exp 22's `ol_exhibits` cell is observed with exp 20's structure-aware measure (the NLA reconstructor's
rebuild of each item's layer-42 vector from Oracle Lens's own free readout, scored by the frozen probe) instead of the
judge, does the exp 22 scorecard move the way exp 19's did?

**Why.** Exp 20 showed on exp 19's bank that both readers' free readouts carry the structural state through the critic
(Oracle Lens 0.70 to 0.94 on the six readable rows, NLA 0.93 to 0.99) while the judge scored exhibition at 0.00 on every
natural row; re-scoring exp 19's `ol_exhibits` cell with that measure at the 0.70 line took the cell from three of eight
to seven of eight and the primary scorecard to 35 of 38 (exp 20, "What this changes"). Exp 22 rebuilt exp 19's bank from
public sources and replicated it (primary 30 of 38; 26 of 30 without the judged exhibits cell), but its `ol_exhibits`
cell is still judge-scored and the judge is blind to structure (0.00 on all seven natural rows, 0.78 on the `code_lang`
positive control; exp 22, reading 5: "the reconstructor-based measure was not run on this bank"). This experiment runs
exp 20's protocol on exp 22's rows and rescores the exp 22 scorecard with it, so the public-bank result is stated with
the same exhibits measure as the exp 19 one.

**Pipeline** (`run25.sh`; app `n7qe`, `modal25.py`). Exp 22's selection split (160 items per row, eight rows including
`code_lang`), its true layer-42 activations (`acts22_{row}.pt`), marked value and labels exactly as exp 22's `_sel`.
Stage `ol` (H100, `b8fn`): Oracle Lens free readout, k = 1, layer 42, WSBench sampling -> `ol_sel_{row}.json`. Stage
`recon` (H200, `m4ks`): NLA explanation k = 1 -> `nla_sel_{row}.json`; the critic's rebuild from the true last 32 and
128 tokens (`span_32`, `span_128`), the same 32 tokens shuffled (seed 25, `bag_32`), the Oracle Lens readout
(`ol_readout`) and the NLA explanation (`nla_readout`), each scored by exp 14's frozen-probe AUROC against the true
activations plus class-direction cosine -> `recon25_{row}.json`. The window sweep and the model-window activations of
exp 20 are not repeated. `analyze25.py` writes `results/table25.md` and calls `rescore25.py`, which replaces the
observed value of exp 22's `ol_exhibits` cell by `ol_readout` at the 0.70 line (predictions, every other cell and every
other threshold unchanged) and writes `results/scorecard25.md` and `results/score25.json`. Fresh readouts: exp 22's
`gen_*.json` are k = 8 on the mining split and are not used.

## Predictions written before the run

Exp 22's span instrument (frozen probe on the critic's rebuild of the true 32-token span, 0.80 line) found five rows
span-sayable: py_string 0.84, latex_math 0.85, py_arg 0.88, json_depth 0.92, code_lang 0.99 (js_string 0.77,
md_heading 0.62, md_list 0.50 are not). Exp 22's `ol_exhibits` predictions already encode this (H on those five, L on
the other three).

1. The critic rebuilds the state from Oracle Lens's k = 1 readout at >= 0.70 on the span-sayable rows py_string,
   latex_math, py_arg and json_depth, so the four natural predicted-high rows flip from MISS (judge 0.00 to 0.03) to ok.
2. The two Markdown rows (md_heading, md_list) stay below 0.70 (exp 20: 0.59 and 0.62 on exp 19's rows; not linearly
   readable at layer 42), so their predicted-low cells stay ok.
3. code_lang stays high (the judge already scored it 0.78; the language is carried by the readout's vocabulary).

If all three hold the primary scorecard goes from 30 of 38 to 34 of 38. The one row the rule leaves exposed is
js_string: predicted low (span 0.77, under the line by 0.03) while exp 20 measured Oracle Lens's readout on exp 19's
js_string row at 0.90; if the readout carries it here too, that cell becomes a LH miss, the same kind as exp 22's
prompted js_string miss. Stated here before the run; the prediction is not changed.

## Results (27 Oct, 18:52 UTC; `results/table25.md`, `results/scorecard25.md`, `results/recon25_{row}.json`; 160 selection items per row; one H100 pass of 14 minutes and one H200 pass of 16 minutes, about $3)

Frozen-probe AUROC against the true layer-42 activations, through the NLA critic:

| row | true span 32 | same tokens shuffled | Oracle Lens readout | NLA explanation | exp 20 on exp 19's row (Oracle Lens) |
|---|---:|---:|---:|---:|---:|
| py_string | 0.84 | 0.46 | 0.86 | 0.99 | 0.91 |
| js_string | 0.77 | 0.46 | 0.84 | 0.97 | 0.90 |
| latex_math | 0.85 | 0.50 | 0.87 | 1.00 | 0.87 |
| py_arg | 0.88 | 0.49 | 0.51 | 0.93 | 0.70 |
| json_depth | 0.92 | 0.76 | 0.97 | 0.99 | 0.94 |
| md_heading | 0.62 | 0.55 | 0.65 | 0.70 | 0.59 |
| md_list | 0.50 | 0.44 | 0.48 | 0.49 | 0.62 |
| code_lang (positive control) | 0.99 | 0.98 | 0.98 | 0.98 | - |

Class-direction cosines: Oracle Lens readout 0.54 to 0.68 on the string, math and depth rows, 0.36 on call argument;
NLA explanation 0.78 to 0.90 everywhere (in-distribution for the critic; see exp 24).

### Scoring the predictions

| prediction | result |
|---|---|
| 1. Oracle Lens readout >= 0.70 on py_string, latex_math, py_arg, json_depth | 3 of 4: 0.86, 0.87, 0.97; call argument 0.51 (HL miss) |
| 2. Markdown rows below 0.70 | holds: 0.65, 0.48 |
| 3. code_lang stays high | holds: 0.98 |
| the stated exposure: js_string predicted low (span 0.77) | materialised: the readout carries it at 0.84 (LH miss, the same row and direction as exp 22's prompted miss) |

### The exp 22 scorecard with this cell (`results/scorecard25.md`)

| scorecard | exp 22, judge-scored exhibits | with the reconstructor measure |
|---|---|---|
| primary (free-readout cells, J-lens on the six uncued rows) | 30 of 38, 0.79 [0.64, 0.89] | 32 of 38, 0.84 [0.70, 0.93] |
| secondary (prompted forced choice) | 13 of 16, 0.81 | 13 of 16, 0.81 (unchanged) |
| all 56 cells, exp 18's convention | 44 of 56, 0.79 [0.66, 0.87] | 46 of 56, 0.82 [0.70, 0.90] |
| without the positive-control row | primary 27 of 34, secondary 11 of 14 | primary 29 of 34, secondary 11 of 14 |

The `ol_exhibits` rule ("Oracle Lens exhibits iff span-sayable") scores 6 of 8 by this measure (4 of 8 by the judge):
HH on py_string, latex_math, json_depth, code_lang; LL on both Markdown rows; LH on js_string; HL on py_arg.

## Reading

1. **Exp 20's exhibits numbers replicate on an independent public bank.** Oracle Lens's readout, which names none of
   these states (exp 22's judge: 0.00 on every natural row), rebuilds string, math and depth state at 0.84 to 0.97
   against 0.87 to 0.94 on exp 19's rows. The one weak row is the same both times: call argument (0.51 here, 0.70
   in exp 20, 0.61 through the second decoder in exp 24). A continuation after a call's third argument does not
   look different enough from one after its first for the critic to tell; the state is in the window (span 0.88)
   and not in what Oracle Lens writes.
2. **Language is vocabulary, structure is arrangement.** The shuffled span keeps code language at 0.98 and drops
   every structural row to 0.44 to 0.50 (JSON depth 0.76, where the vocabulary of deep and top-level strings differs,
   exp 22's lexical-cue note). This is exp 24's point in its cleanest form, on one bank with one decoder.
3. **The span instrument bounds exhibition loosely.** JavaScript string sits under the 0.80 line (0.77) and the
   readout carries it at 0.84; call argument sits above (0.88) and the readout carries it at 0.51. The rule's two
   misses here are the two directions a floor-type instrument can miss in, and the LH miss is on the row exp 22's
   prompted cell also missed. With both measures the rule stands at 6 of 8 on this bank and 7 of 8 on exp 19's.
4. **The scorecard clears the declared bar with the reconstructor measure.** Primary 32 of 38 (0.84), all cells
   0.82 against the pre-registered 0.80; the judge-scored version (0.79) stays in the record as the pre-registered
   measure, and the replacement is post hoc and disclosed, as in exp 20's re-scoring of exp 19 (35 of 38 there).
   The secondary scorecard is untouched. Both versions are quoted together wherever the number appears.

## What this changes

Exp 22's primary scorecard is quoted as 30 of 38 by the pre-registered judge measure and 32 of 38 by the
reconstructor measure. The thesis sentence is unchanged: the readers carry structural state in their outputs and
do not name it; exps 24 and 25 together put that on two decoders and two prospective banks.
