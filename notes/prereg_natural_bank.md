# Pre-registration: the prediction test on a natural-text bank, with the revised NLA rules (exp 19)

Drafted 2026-10-27 (Suraj, with assistant). Status: DRAFT until the instruments have run and the prediction cells are
filled; then this file's sha256 goes into `notes/decisions.md` and the readers run. Nothing below changes a frozen
phrasing, direction, dose or seed of exps 13 to 18. Private until shared with Guiv.

## 1. What changed since exp 18 and why

Exp 18 scored 0.63 over 70 cells against a declared 0.80. Three of its misses have a named cause that this test
removes or re-states; the rest of the design is carried over unchanged.

1. **The banks were synthetic and lexically cued.** Every exp 18 row was generated from templates, so the two values
   differed in vocabulary and a bag-of-tokens probe on the text alone could tell them apart. That made the J-lens
   cells (2 of 10) uninterpretable and left a question over the "readable" column. Here every item is a read point
   inside a real file from a public source (pip and npm package sources on this machine, arXiv e-prints), and the two
   values of a row come from the same files. A bag-of-words probe on the alphabetic word tokens among the last 48 tokens (the structural marker tokens a row
   is defined by do not count; grouped 5-fold by source file, `lib19.lexical_cue_auroc`) is run at bank time; a row at or above **0.65** is kept in the bank but its J-lens
   cell is not scored.
2. **"NLA states iff used" failed (5 of 10).** The NLA stated LaTeX-math never (the most used row) and language 95%
   (unused). The revised rule (D42) is that the NLA states what is statement-sayable: the reconstructor shift along
   the class direction when fed the fixed statement pair, ratio ≥ 0.30. Post hoc on exp 18 this rule scores 8 of 10
   (misses: json_depth 0.25 and call_arg 0.33 observed stated with stmt 0.13 and 0.09). Reported here as disclosure;
   it is a prediction only for the rows below.
3. **"NLA never names the complement" failed (4 of 10).** It named both values on every ordinal row (first/third,
   first/fourth, shallow/deep, level one/three) and stayed one-sided only where the unmarked value is an absence
   (outside a string, outside math, outside a quote). The revised rule: the NLA names the complement iff both values
   are positive states (`both_positive` flag, declared per row below). Post hoc on exp 18: 10 of 10.

Not changed: the Oracle Lens rules (9 of 10 and 10 of 10 in exp 18), the "exhibits" rule (6 of 10; the judge cannot
see structural exhibition, kept as is and reported separately), the J-lens rule (jenr OR stmt, now on a bank where
the lexical cue is measured), thresholds, dose, nulls, layers, the mining and selection procedure, the judge.

## 2. Instruments (unchanged from exp 18 §2)

| property | instrument | threshold |
|---|---|---|
| readable | grouped 5-fold linear probe on the 160 selection items, true activations (`mats.probes.cv_auroc`) | AUROC ≥ 0.80 |
| sayable by span | probe fit on true activations, applied frozen to the NLA reconstructor's rebuild of the item's last 32 tokens (`lib19.frozen_probe_auroc`) | AUROC ≥ 0.80 |
| sayable by statement | reconstructor shift along the true class direction for the fixed statement pair, as a fraction of the class gap | ratio ≥ 0.30 |
| used | centred class-mean push at r11 at the read token, next-token log-odds on the row's declared token set, 20 isotropic nulls, two-sided | \|z\| ≥ 3 |
| J-enriched | enrichment of the class-mean direction in the rank-1536 J-space | ≥ 1.5 |
| lexical cue (bank admission) | bag-of-words probe on the alphabetic word tokens among the last 48 tokens, grouped by source file | < 0.65 for the J-lens cell to count |

## 3. Prediction rules

| reader output | predicted high iff | counts as high |
|---|---|---|
| NLA prompted AUROC (fixed label + 3 mined phrasings, mean) | readable | ≥ 0.85 |
| NLA states the property (judge, 40 items × 8 samples) | **statement-sayable** | ≥ 20% of readouts |
| NLA names the complement when asked | **both values are positive states** (`both_positive`) | fixed-pair argmax on the unmarked value ≥ 20% |
| Oracle Lens prompted AUROC | sayable by span | ≥ 0.70 |
| Oracle Lens exhibits the value in free text | sayable by span AND value ≤ 2 tokens | ≥ 30% |
| Oracle Lens states the property | never | ≥ 5% |
| J-lens prompted AUROC | J-enriched OR statement-sayable; scored only on rows with lexical cue < 0.65 | ≥ 0.70 |

## 4. Rows (eight; two values each, from the same files; `experiments/19_natural_bank/feature_spec19.py`)

| row | values (marked first) | source | next-token set | constrains next | ≤ 2 tokens | both positive |
|---|---|---|---|---|---|---|
| py_string | inside / outside a string literal | .py files | " ' | yes | yes | no |
| js_string | inside / outside a string literal | .js files | " ' | yes | yes | no |
| latex_math | inside / outside an inline math span | arXiv .tex | $ | yes | yes | no |
| py_arg | third / first positional argument | .py files (ast) | , ) | yes | yes | yes |
| json_depth | three or more objects deep / top level | .json files | } | yes | yes | yes |
| html_depth | six or more open elements / at most three | .html files | </ | yes | yes | yes |
| md_heading | level 3-4 / level 1-2 heading | .md files | none | no | yes | yes |
| md_list | indented / top-level list item | .md files | none | no | yes | yes |

A CSV-field row was planned and dropped at build time: the public CSV files on this machine are numeric test
fixtures and yielded 16 items. Bank: 200 items per row (20 + 20 mining, 80 + 80 selection), clustered by source file; built by `build19.py`
(seed 1919), sha256 and the lexical-cue column filled in §6 before any reader runs.

## 5. Scoring rule (unchanged)

A cell is correct when the observed value lands on the declared side of its threshold. Accuracy over all scored cells
with a Wilson 95% CI and the confusion table per reader; the account is predictive at ≥ 0.80. The two revised NLA
rules are also reported on their own (18 cells) so that a pass or fail of the revision is visible apart from the
carried-over rules.

## 6. Bank and instruments (filled before the readers run)

(pending)

## 7. Budget

One GPU at a time; about $10 GPU (instruments, generation, scoring at the exp 18 rates), about $3 judge and mining.
