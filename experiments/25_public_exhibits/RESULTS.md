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

## Results

(pending: `results/table25.md`, `results/scorecard25.md`)
