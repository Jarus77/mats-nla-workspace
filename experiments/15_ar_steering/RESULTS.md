# 15 — AR-mediated steering: who follows which direction (run 25 Oct 2026)

**Question.** Does Qwen3.6-27B *use* the table-column and parenthesis state the NLA reports, and can the
reconstructor (the NLA's AR) build a working steering vector for that state from text, and from a
statement that merely names it? Which reader's readout then tracks what the model uses?

**Design (spec15.py).** Layer 42, 200 items per concept (100 / 100; exp 13 banks). Eight directions:
class mean (`cm`), and NLA-AR class means of NLA explanations (`arexpl`), of the true 32-token spans
(`arspan`) and of a fixed statement pair (`arstmt`), each raw and with the J-space (4r* = 1536) removed.
Centred edit h − (v·h)v + s·dose·v at the read position, dose r11 = 0.11‖h‖ (same norm for every
direction) and q90 (direction's own 90th-percentile projection). Nulls: 20 isotropic + 20 J-complement
random directions at r11. Model side: next-token log-odds of the newline set (table column) and of the
")" set (parenthesis), plus a 24-token greedy continuation. Readers: NLA, OLens@42, J-lens@42 prompted
log-odds on the frozen fixed label + 3 mined phrasings (B − A, + = toward `values[1]`). Statistic: paired
± mean shift, z against the null band. 113 conditions × 200 items per concept.

## Headline table (r11, z against the 20 isotropic nulls; + = the expected direction)

**Table column** (model = newline log-odds, + = "late column")

| direction | model | NLA | OL | J-lens |
|---|---:|---:|---:|---:|
| class mean | +7.1 | +13.2 | +3.0 | +2.2 |
| class mean, J-removed | +5.0 | +9.4 | +2.3 | +0.9 |
| AR(NLA explanations) | +8.6 | +14.1 | +4.2 | +1.9 |
| AR(explanations), J-removed | +6.2 | +9.8 | +2.9 | +0.9 |
| AR(32-token spans) | +10.5 | +5.8 | +2.4 | +0.3 |
| AR(spans), J-removed | +6.1 | +4.2 | +1.0 | +1.0 |
| **AR(fixed statement)** | **+0.7** | +4.6 | **+12.3** | **+10.1** |
| AR(statement), J-removed | +0.1 | +2.7 | +4.0 | +1.4 |

**Parenthesis** (model = ")" log-odds, sign flipped so + = "outside" lowers ")")

| direction | model | NLA | OL | J-lens |
|---|---:|---:|---:|---:|
| class mean | +19.8 | +6.3 | +10.4 | 0.0 |
| class mean, J-removed | +8.2 | +3.9 | +7.9 | −0.1 |
| AR(NLA explanations) | +12.9 | +26.3 | +13.6 | −0.1 |
| AR(explanations), J-removed | +5.7 | +23.7 | +11.5 | −0.1 |
| AR(32-token spans) | +6.0 | +15.0 | +8.6 | −1.1 |
| AR(spans), J-removed | +2.3 | +13.1 | +7.3 | 0.0 |
| **AR(fixed statement)** | **−3.6** | +2.6 | +3.6 | −1.0 |
| AR(statement), J-removed | −2.7 | −0.6 | +2.0 | −0.2 |

The J-complement null band gives the same picture (e.g. table column, statement direction: model +1.1,
NLA +4.8, OL +15.7). Mean shifts in nats at r11: class mean moves the newline log-odds +0.38 and the
")" log-odds −0.68; the NLA mean-phrasing readout moves +3.4 / +1.0; OL +0.8 (statement) / +1.8 (class
mean). Figure: `results/fig15_who_follows.png`.

## What it says

1. **The model uses the state, and the reconstructor already builds working steering vectors from real
   text.** Directions built by the NLA-AR from NLA explanations or from the true 32-token spans move the
   model's next-token behaviour as much as the class-mean direction (table column: span direction z
   +10.5 > class mean +7.1; parenthesis: +12.9 and +6.0 vs +19.8), with the J-space removed as well. For
   self-inducing concepts the "AR as steering-vector generator" question (Kylie / Neel, drop-in) has a
   positive answer.
2. **The statement-built direction is a clean dissociation.** AR(fixed statement) has cosine 0.10 / −0.04
   with the class mean, separates natural activations at AUROC 0.85 / 0.39, and moves the model's
   behaviour not at all (table column) or the wrong way (parenthesis). On table column it is nonetheless
   the direction OLens (+12.3) and the J-lens (+10.1) respond to most strongly, and the NLA responds to
   mildly (+4.6). A direction that is J-readable and OL-sayable but not used.
3. **Reader sensitivity does not track model use, and the NLA over-reads its own reconstructor's
   direction.** On parenthesis the NLA responds four times more to AR(explanations) (+26.3) than to the
   class mean (+6.3) while the model responds less to it (+12.9 vs +19.8). The three AR directions are
   built with the NLA's own AR, so part of the NLA's response is circular (the direction is "what the NLA
   writes about"); the class-mean rows are the uncontaminated comparison. OLens on parenthesis follows the
   used directions in rough proportion (+10.4 / +13.6 / +8.6) and the statement direction weakly (+3.6).
   On table column OLens barely follows any used direction (+2.3 to +4.2) and follows the unused lexical
   one strongly.
4. **J-lens reads nothing on parenthesis under any push** (|z| ≤ 1.1), consistent with parenthesis state
   lying outside the J-space; on table column it reads only the statement direction.

**Predictions (README) vs outcome.** "Class-mean and AR-from-exhibited-text vectors move the model and
the NLA readout": confirmed. "AR-from-statement moves nothing": true for the model, **false for OLens
and the J-lens on table column**. "OLens moves only through continuation": false, its prompted
readout follows the used directions on parenthesis. "J-lens flat": true except the statement direction on
table column.

## Geometry (prep)

| | table column | parenthesis |
|---|---|---|
| cos(class mean, AR explanations / spans / statement) | 0.84 / 0.31 / 0.10 | 0.89 / 0.51 / −0.04 |
| natural AUROC of h·v (cm / arexpl / arspan / arstmt) | 1.00 / 1.00 / 0.97 / 0.85 | 0.96 / 0.97 / 0.88 / 0.39 |
| J-enrichment at r1536 (cm / arexpl / arspan / arstmt) | 0.98 / 0.95 / 0.78 / 1.12 | 1.20 / 1.41 / 1.24 / 1.06 |
| NLA-AR reconstruction FVE (explanations / spans) | 0.52 / −0.65 | 0.75 / −0.01 |
| probe AUROC on reconstructions (true / arexpl / arspan) | 1.00 / 1.00 / 1.00 | 0.92 / 0.91 / 0.87 |
| q90 dose as fraction of median ‖h‖ (cm / arstmt) | 0.19 / 0.03 | 0.15 / 0.03 |

The span reconstructions have negative FVE yet their class-mean direction is the best table-column
steering direction: the AR gets the per-item vector wrong and the class difference right. q90 doses are
direction-specific and tiny for the statement direction, so r11 (equal norm) is the comparison to cite;
q90 cells are in the JSON.

**Zero-dose readout AUROC vs class** (sanity that the readers see the concept unsteered): NLA 0.998 /
0.96 (mean phrasing), OLens 0.72 / 0.70, J-lens 0.64 / 0.47, model next-token log-odds 0.70 / 0.92.

## Null result to record: the greedy-continuation metrics do not move

Cells-to-newline is identical on every item under every condition (null band sd 0.000): the model
always completes the five-column row, a one-position patch changes which words it writes, not how many
cells. Tokens-to-close moves within noise (shift ≤ 0.03 vs null sd 0.03; found rate 0.52). The newline
log-odds start near −13, so a +0.4 to +0.7 nat push never flips the argmax. The paired next-token log-odds
are the behavioural measure for this experiment. Sanity: zero-dose equals no-hook (table column exactly,
parenthesis 0.04 tokens); unsteered class means separate (5.5 vs 1.8 cells; 3.1 vs 23.6 tokens).

## Caveats

- Single layer and position (42, read position); single equal-norm dose r11; 20 + 20 nulls; prompted
  readouts only (no free-text judging here; exp 12/14 cover that).
- The three AR directions all come from the NLA's reconstructor; OL's own AR (exp 14) writes in a
  whitened space whose moments are gated, so an OL-AR-built direction was not possible.
- OLens is off its trained grid at 42 (exp 11 point 1: 40/44 behave alike on these items).
- The statement direction's lexical content is a hypothesis for why OL and J-lens follow it; a token-level
  readout of that direction (unembedding top-k) would confirm it and is not done here.

## Run record

Modal app `vx4`, one GPU at a time (`RFM_MAXC=1`), 25 Oct 04:00–08:00 IST. prep ≈ 15 min H200; behave 4
chunks 13 + 12 + 23 + 22 min H100; score_nla 71 + 72 min H200; score_ol 37 + 35 min H100; analyze CPU.
≈ 2.7 H200-h + 2.4 H100-h ≈ $22. Two fixes during the run: Modal function names made opaque (54bf93a);
the frozen-template file shipped into the image (d3b63ed, the score stages had failed on a repo-relative
path). A five-minute timing probe (`--stage probe`) and an optional scoring trim (`RFM_SCORE_T`,
`RFM_SCORE_NULLK`, unused) were added (7125ebb). Outputs: `rfm-results:/15_ar_steering/` (spec, 4 behave,
4 score, 2 analysis files; copies of the analysis JSON and the figure in `results/`). Report:
`python report15.py --analysis results/analysis15_tb_tablecol.json results/analysis15_tb_paren.json`.
