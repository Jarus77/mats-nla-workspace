# Pre-registration: does a reader's training objective predict its blind spots? (week-two prediction test)

Drafted 2026-10-26 (Suraj, with assistant). Status: DRAFT until the instruments have run and the prediction
cells are filled; then this file's sha256 goes into `notes/decisions.md` and the readers run. Nothing below
changes a frozen phrasing, direction, dose or seed of exps 13 to 17. Private until shared with Guiv.

## 1. Claim under test

For a concept at layer 42, four properties measured without running any reader (readable, sayable by a
32-token span, sayable by a statement, used by the model) predict each reader's prompted score and
spontaneous behaviour, before the reader is run. The account was built on two position-in-text rows
(exps 13 to 17); this test asks whether it holds out of sample on 12 to 15 new concepts.

## 2. Instruments (existing code; one threshold each, fixed here)

| property | instrument | function | threshold |
|---|---|---|---|
| readable | grouped 5-fold linear probe on the 160 selection items, true activations | `mats.probes.cv_auroc(X, y, groups, k=5)` | readable if AUROC ≥ 0.80 |
| sayable by span | a probe fit on the true activations (grouped 5-fold) applied frozen to the NLA reconstructor's rebuild of each item's true last 32 tokens, exp 14's headline measure | `lib18.frozen_probe_auroc(true, nla.reconstruct(spans), y, groups)` | sayable if AUROC ≥ 0.80. **Clarified 26 Oct before any reader ran, after the first instrument run:** the gap-ratio form first written here (`ar_lib.gap_retention` ratio ≥ 0.50) was uninformative on these rows (0.02 to 0.12 everywhere, 0.38 on stance, the NLA reconstructor shrinks every span rebuild), so the instrument is the frozen-probe AUROC that exp 14 actually reported; the gap ratio is kept in the table as `span_gap` for the record. |
| sayable by statement | shift of the NLA reconstructor's output along the true class direction when fed the fixed-label statement pair | `ar_lib.statement_shift(true, y, rec_a, rec_b)["ratio"]` | statement-sayable if ratio ≥ 0.30 |
| used | centred class-mean push at r11 (0.11‖h‖) at the read token, next-token log-odds on the row's declared token set, 20 isotropic nulls | exp 18 `lib18.centred_push` + `band` (the exp 15 construction); model side only | used if \|z\| ≥ 3 against the isotropic band, two-sided; the sign of the shift is recorded, not predicted (clarified 26 Oct after the first instrument run showed a −12 cell; no reader had run) |
| J-enriched | enrichment of the class-mean direction in the rank-1536 J-space | `mats.jspace.enrichment(basis, v, 1536)` | J-enriched if ≥ 1.5 |

The model-side token set for "used" is declared per concept in §4 before the push is run. Rows whose next
token is not a discrete decision (a count with no token consequence) get the set "none" and are scored
"not used" by construction; this is a prediction, not a gap.

## 3. Prediction rules (one row per reader × output; a cell is "predicted high" or "predicted low")

| reader output | predicted high iff | threshold that counts as high |
|---|---|---|
| NLA prompted AUROC (frozen fixed label + 3 mined phrasings, mean) | readable | ≥ 0.85 |
| NLA states the property spontaneously (judge, 40 items × 8 samples) | used AND the concept constrains the next token (declared flag in §4) | ≥ 20% of readouts |
| NLA names the complement (the unmarked value) when asked | never (predicted low for every row, from exp 17) | multi-candidate argmax on the unmarked value ≥ 20% |
| Oracle Lens prompted AUROC | sayable by span | ≥ 0.70 |
| Oracle Lens exhibits the value in its free text | sayable by span AND the value is expressible in ≤ 2 tokens (declared flag) | ≥ 30% of readouts |
| Oracle Lens states the property | never (predicted 0% for every row) | ≥ 5% |
| J-lens prompted AUROC | J-enriched OR statement-sayable | ≥ 0.70 |

Each rule is falsifiable: a row where the antecedent holds and the cell lands low, or the antecedent fails and
the cell lands high, is a miss.

## 4. Concept list (12 to 15 rows; fill the four instrument columns and the two flags BEFORE any reader runs)

**Bank status (26 Oct).** Rows 1 to 7, 13, 14 and 15 are built and frozen: `experiments/18_prediction_test/`
(`items_build18.py`, tested; `modal18.py --stage build`), 200 items per row (20 + 20 mining, 80 + 80
selection), `/results/18_prediction_test/items_base.json`, sha256
`44308dfdc1b92898b2f463add7117d0fdd15f6a8a626b179ccd68ae7c9efcd36`, 2,000 items. Token length is matched
between the two values within 20% on every synthetic row (medians in `results/build_info.json`); the quote
control (exp 07's generator, band 40 to 250 tokens) keeps a residual gap (105 vs 136), so its prompted cells
are scored after exp 13's length-control regression. Rows 8 to 11 wait on Guiv's bank; row 12 (document
topic) needs a topic-labelled passage source and is deferred. Row names in the bank: json_depth,
latex_math, py_string, call_arg, md_heading, csv_col, html_depth, stance_word, ij2_lang, bk_quote.

Generators follow exp 13's `items_build` style (true value known, 20 + 20 mining and 80 + 80 selection items,
clustered by passage, values length-matched where possible). Rows marked G come from Guiv's bank once his
files arrive and are scored with our instruments first.

| # | concept | values | generator | next-token set for "used" | constrains next token? | value ≤ 2 tokens? | readable | span | stmt | used | J-enr | predicted NLA / OL / J-lens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | JSON nesting depth at the read key | depth 1 vs 3 | new, JSON objects | `}` set | yes | yes | | | | | | |
| 2 | inside a LaTeX math environment | inside `$...$` vs prose | new | `$` set | yes | yes | | | | | | |
| 3 | inside a string literal (Python) | inside vs outside quotes in code | new | quote-close set | yes | yes | | | | | | |
| 4 | positional argument index in a call | 1st vs 3rd argument | new | `,` / `)` set | yes | yes | | | | | | |
| 5 | markdown heading level | H1 vs H3 | new | none | no | yes | | | | | | |
| 6 | CSV column index | 1st vs 4th field | new | `,` / newline set | yes | yes | | | | | | |
| 7 | unclosed HTML tag depth | 1 vs 3 open tags | new | `</` set | yes | yes | | | | | | |
| 8 | G: code language | Python vs JS | Guiv / exp 07 `build_code` | none | no | yes | | | | | | |
| 9 | G: JSON key vs value position | key vs value | Guiv | `:` set | yes | yes | | | | | | |
| 10 | G: LLM-mined row 1 | from Guiv's mined set | Guiv | declared on arrival | | | | | | | | |
| 11 | G: LLM-mined row 2 | from Guiv's mined set | Guiv | declared on arrival | | | | | | | | |
| 12 | sayable-but-unused candidate: document title topic | sports vs finance passage | new, topic by source | none | no | yes | | | | | | |
| 13 | sayable-but-unused candidate: author stance word present | "unfortunately" vs "fortunately" earlier in the passage | new | none | no | yes | | | | | | |
| 14 | control: language | FR vs DE | exp 07 `build_lang` | none | no | yes | | | | | | |
| 15 | control: quote state | inside vs outside quotes | exp 07 `build_quote` | quote-close set | yes | yes | | | | | | |

Rows 12 and 13 are chosen to be statement-sayable and J-enriched but not used at the read token, so the
account can fail on them (prediction: OL and J-lens high, NLA states low). Rows 14 and 15 are in-workspace
controls (prediction: every reader high).

## 5. Scoring rule

A cell counts as correct when the observed value lands on the declared side of its threshold. Report
accuracy over all cells (≥ 36 cells: 12 rows × 3 prompted scores at minimum, plus the spontaneous cells),
with a Wilson 95% CI and the full confusion table (predicted high / low × observed high / low) per reader.
The account is "predictive" if accuracy ≥ 0.80. Readers that saturate (every row high) are reported but do
not count toward the claim. A row whose instruments disagree with its intended role (e.g. row 12 turns out
used) keeps its instrument-based prediction; the role note is for reading only.

## 6. Procedure and hash

1. Build items for the new rows; freeze the item banks (sha256 in this file). Done 26 Oct for ten rows (§4).
2. Run the instruments (one GPU job: probe, NLA-reconstructor span and statement passes, model-side push,
   enrichment). No reader is run in this job.
3. Fill the instrument columns and the predicted cells in §4. Record this file's sha256 in `decisions.md`.
4. Run the readers: prompted log-odds on frozen phrasings (mined on the mining split with exp 07's
   procedure, frozen before scoring), spontaneous judge on 40 × 8 free readouts, the multi-candidate argmax
   for the complement check. Oracle Lens also at layers 36, 44, 52 (prompted only) for the sayable-vs-readable
   contrast across layers.
5. Score §5; write the row-by-row table and the confusion tables into `experiments/18_prediction_test/RESULTS.md`.

## 7. Budget and timing

At most two GPU-days, cost ceiling $40, one GPU at a time on the main workspace. Day 1 items and
instruments; day 2 predictions filled and readers run; day 3 scoring and write-up.
