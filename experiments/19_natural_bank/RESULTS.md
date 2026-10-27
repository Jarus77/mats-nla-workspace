# Exp 19: the prediction test on a natural-text bank, with the revised NLA rules

Pre-registration: `notes/prereg_natural_bank.md` (rules written 27 Oct before any instrument ran; predictions filled
and hashed before any reader ran). Pipeline: exp 18's, cloned (`modal19.py`, app `k3rw`); the bank is built on the
laptop by `build19.py` from public package sources on this machine and fresh arXiv e-prints, and uploaded.

## Bank (27 Oct)

Eight rows, 200 read points each (20 + 20 mining, 80 + 80 selection), both values of a row from the same source
files, clustered by file. sha256 `7ddfbceac1a5510156b1037239606536e50484a61f93c057629c26fa41525460`. A CSV row was
dropped at build time (16 items: the CSV files here are numeric fixtures).

| row | source files | lexical cue (bag of word tokens, last 48) | J-lens cell scored |
|---|---|---|---|
| py_string | 196 .py | 0.47 | yes |
| js_string | 193 .js | 0.56 | yes |
| latex_math | 157 arXiv .tex | 0.56 | yes |
| py_arg | 192 .py | 0.49 | yes |
| json_depth | 184 .json | 0.94 | no (deep and top-level strings use different words) |
| html_depth | 194 .html | 1.00 | no (deep text is navigation and list items, shallow is paragraphs) |
| md_heading | 189 .md | 0.45 | yes |
| md_list | 179 .md | 0.52 | yes |

## Instruments (27 Oct, before any reader ran; `results/predictions19.md`, hashes in `notes/prereg_natural_bank.md` §6)

| row | readable | span (frozen probe) | statement | used z | J-enr |
|---|---:|---:|---:|---:|---:|
| py_string | 1.00 | 0.85 | 0.04 | +6.6 | 1.17 |
| js_string | 0.99 | 0.82 | 0.01 | +0.7 | 1.22 |
| latex_math | 0.99 | 0.83 | 0.05 | +25.6 | 1.31 |
| py_arg | 0.99 | 0.89 | 0.23 | +3.2 | 1.10 |
| json_depth | 0.99 | 0.87 | 0.14 | -2.0 | 1.29 |
| html_depth | 0.99 | 0.74 | 0.04 | -4.6 | 1.32 |
| md_heading | 0.63 | 0.51 | 0.03 | n/a | 1.48 |
| md_list | 0.65 | 0.64 | 0.07 | n/a | 1.42 |

Two things the synthetic banks of exp 18 hid: heading level and list nesting are not linearly readable at layer 42 on
real Markdown (0.63, 0.65; synthetic 1.00), and no natural row is statement-sayable (max 0.23), so the revised rule
predicts the NLA states nothing anywhere on this bank.

## Scorecard (`results/scorecard19.md`, `results/score19.json`; readers at layer 42, Oracle Lens also at 36/44/52)

**Primary scorecard (free-readout and reconstructor-based cells, J-lens on the six uncued rows): 31 of 38, 0.82
[0.67, 0.91].** Without the Oracle Lens "exhibits" cell, whose judge again scored 0.00 on every row (see below): 28 of
30, 0.93 [0.79, 0.98]. **Secondary (prompted forced-choice cells): 11 of 16, 0.69 [0.44, 0.86].** All 56 cells
together, the exp 18 convention: 44 of 56, 0.79 [0.66, 0.87], against the declared 0.80.

| rule | score | what happened |
|---|---|---|
| NLA states iff statement-sayable (revised) | 8 / 8 | no row statement-sayable, no row stated (max 0.19); the exp 18 rule "states iff used" would have missed four rows here |
| NLA names the complement iff both values are positive states (revised) | 6 / 8 | named on py_arg 0.74, json 0.66, html 0.94, heading 1.00; misses: latex_math 0.41 (an absence row, named) and md_list 0.00 (both-positive, not named; the row is not readable) |
| Oracle Lens never states | 8 / 8 | max 3% |
| Oracle Lens exhibits iff span-sayable | 3 / 8 | 0.00 on every row again: the judge does not see structural exhibition; exp 20 replaces this cell with a reconstructor-based measure |
| J-lens high iff J-enriched or statement-sayable | 6 / 6 | all predicted low, all low (0.40 to 0.68); the two lexically cued rows (json 0.68, html 0.62) excluded as declared |
| NLA prompted high iff readable (secondary) | 5 / 8 | three rows just under 0.85 (0.84, 0.84, 0.79); both unreadable rows low as predicted |
| Oracle Lens prompted high iff span-sayable (secondary) | 6 / 8 | misses in the direction that hurts the rule: py_string 0.52 and json_depth 0.64 with span 0.85 and 0.87; flat across layers 36 to 52 on every row |

Confusion, predicted x observed: NLA HH 7, HL 4, LH 1, LL 12; Oracle Lens HH 3, HL 7, LH 0, LL 14; J-lens LL 8.

## Reading

1. **The revised NLA rules pass on natural text (14 of 16).** The NLA's spontaneous statements track the reconstructor's
   statement shift and not the model's use of the state: LaTeX math is the most used row (z +25.6) and is never stated;
   nothing with a statement shift under 0.30 is stated. Complement naming follows the positive-state flag on six of
   eight rows.
2. **Oracle Lens never states a property and the lens is flat across its trained layers**, as in exps 12, 13 and 18.
3. **Span-sayability bounds Oracle Lens detection but does not set it.** No row is detected without being span-sayable
   (LH 0), but two span-sayable rows are not detected (py_string 0.52, json_depth 0.64). This is the direction Guiv's
   E45 pointed at: the span re-induces the state and the lens still misses it. On both rows the fixed phrasing scores
   reversed (py_string 0.31, json_depth 0.33 at layer 42; `templates_frozen.json`), so the lens's log-odds do separate the
   classes with the wrong sign, and the three mined phrasings land near chance; the prompted cell is phrasing-bound in a
   way the free-readout cells are not. Exp 20 measures the model's own window map directly, without the NLA critic, on
   the same rows.
4. **The judge cannot score exhibition of structure** (0.00 on all 16 rows across exps 18 and 19); the cell is kept
   for the record and replaced by exp 20's reconstructor-based measure.
5. **The J-lens cells are uninformative here**: nothing is J-enriched, nothing is J-lens-readable, as the Ostrow et al.
   (2026) account of the lens predicts for features that change every token.

## Bank provenance and what is tracked

The bank was built from files on the author's machine outside this repository plus arXiv e-prints. A path audit after
the fact found that the Python, JavaScript and JSON sources are package caches (site-packages, node_modules, uv and bun
caches) and the LaTeX is arXiv, while most Markdown and all HTML sources are local files of unverified provenance
(skill files, documentation copies, cached pages). The bank and the reader generations are therefore not published;
their sha256 values are recorded, the per-item scores and judge verdicts are tracked, and a public-source-only rebuild
is the first thing to do if this experiment goes into the write-up as more than a replication.

Cost: about $9 GPU (instruments 1 H100-hour and 1 H200-hour, readouts and scoring about the same), $1.50 judge and mining.
