# Exp 26: a broad layer sweep on the public bank

**Question.** The mentor asked for layers far apart: "test layers that are far apart ... layer 10 or layer 60 might
be quite different ... do a broad sweep" (23 Oct call; `notes/draft_landscape_2026-10-24.md` B.1, "towards 50 ... a
broad sweep"). Does anything about the readers' picture change with the layer on the x-axis, from 20 to 60?

**What is already known.** Exps 18, 19 and 22 scored Oracle Lens prompted at 36, 42, 44 and 52 as a sensitivity
check: every row is flat within 0.08 across those four layers (exp 18 RESULTS, reading on the OLens layers; exp 19 and
exp 22 `ol_prompted_by_layer` in `score19.json` / `score22.json`, the exp 22 spread 0.03 to 0.07 per row). The NLA
cannot be moved off 42 (its injection is fixed to the layer it was trained on; `notes/decisions.md`, prereg §5), so
this sweep is an Oracle Lens and probe sweep; the NLA enters only as the critic that rebuilds Oracle Lens's readouts.
The Oracle Lens is trained on layers 20 to 60 step 4; 42 is off its grid and is read as exp 22 reads it (the carrier
says "layer 42").

**Pipeline** (`run26.sh`; app `k2vd`, `modal26.py`; stages strictly one after the other). Exp 22's selection split
(160 items per row, eight rows including the `code_lang` positive control), the marked value and the labels exactly as
exp 22's `_sel`. Stage `ol` (H100, `r6tp`): capture the read-token activation at the twelve layers 20, 24, ..., 60 and
42 with the adapter disabled -> `acts26_{row}.pt` (bf16, the model's own precision); per layer the grouped 5-fold probe
AUROC (`readable`), the Oracle Lens prompted AUROC with exp 22's four frozen phrasings of the row (the three mined ones
and the fixed label, `templates_frozen.json`; the cell value is the max of the mean-of-three AUROC and the fixed
label's, exp 22's convention; `ol_prompted`) and the Oracle Lens free readout, k = 1, WSBench sampling ->
`ol26_{row}_L{L}.json`, `sweep26a_{row}.json`. Stage `recon` (H200, `x9ae`): each layer's readouts through the NLA
reconstructor, scored by exp 14's frozen probe against the TRUE layer-42 activations (exp 25's `ol_readout` measure;
`carries`), and the strict naming rate (`names`): exp 23's state regexes (`lib23.STATE_RE`) mapped py_string and
js_string -> in_string, json_depth -> brace_depth, latex_math -> in_math, md_heading -> heading_level, md_list ->
list_depth; py_arg and code_lang have no exp 23 state, so `lib26.NAME_RE` declares py_arg
`\b(first|second|third|1st|2nd|3rd)\s+(argument|parameter|arg)\b` and code_lang `\b(python|javascript|typescript)\b`
(case-insensitive) -> `sweep26_{row}.json` (stage 1's keys plus `carries` and `names`). `analyze26.py` writes
`results/table26.md` (one table per measure, rows x layers), `results/fig26_{row}.png` and the 2 x 4 grid
`results/fig26.png` (x = layer with 42 marked, y = AUROC 0.4 to 1.0, lines at 0.70 and 0.80; the naming rate in a
strip below each panel). Every metric is computed per layer; no per-item best layer is ever taken.

**Cost estimate.** Capture is one pass per row (160 sequences, twelve hooks). Prompted log-odds: 12 layers x 8 rows x
160 items x 8 candidates = 122,880 teacher-forced rows, 64 per forward, about 1,900 forwards of some 40 tokens: a
few minutes. Readouts: 12 x 8 x 160 = 15,360 generations of up to 256 tokens in batches of 32, 480 generate calls;
exp 25's stage `ol` did 40 such calls in about 10 minutes after loading, so about 2 hours on the H100 (the long part;
under the 4-hour function timeout, and each row and each layer's readout file is skipped on a relaunch). Stage `recon`:
96 reconstructor calls of 160 texts, about 25 to 30 minutes on the H200 (exp 25's 40 calls took 16 minutes with the
NLA explanations). Roughly $10 (H100) + $3 (H200), about $13 in all.

## Predictions written before the run

1. The readable probe is near 0.5 at layer 20 and rises to a plateau by about layer 36 on the six readable rows
   (py_string, js_string, latex_math, py_arg, json_depth, code_lang); it stays near 0.6 at every layer on the two
   Markdown rows (md_heading, md_list).
2. Oracle Lens prompted AUROC is flat within 0.10 from 36 to 60 on every row (extending the 36/44/52 finding) and may
   fall at 20 to 28 where the probe falls.
3. The carried measure (Oracle Lens readout through the critic) tracks the probe and not the prompted score: it stays
   at or above 0.70 wherever the probe is at or above 0.80, on the string, math and depth rows (py_string, js_string,
   latex_math, json_depth).
4. The strict naming rate stays at or below 0.02 at every layer on every structural row (the positive control
   code_lang is exempt: its regex is the language name, which the readouts use freely).

## Results

(pending: `results/table26.md`, `results/fig26.png`, `results/sweep26_{row}.json`)
