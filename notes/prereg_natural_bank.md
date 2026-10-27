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

## 5. Scoring rule (revised 27 Oct, before any instrument result was read for html/md rows and before any reader ran)

Two scorecards, declared here because forced-choice scoring puts both readers off distribution (Neel, 27 Oct) and we
agree it is the weakest of the measures.

**Primary (reconstructor- and behaviour-based cells, 5 per row, 40 cells):** NLA states, NLA names the complement,
Oracle Lens exhibits, Oracle Lens states, and the J-lens prompted cell on rows with lexical cue < 0.65. These are the
cells whose antecedents are reconstructor measurements (span, statement) or the declared flags, and whose observed
values come from free readouts judged, or from a probe-free direction. Predictive at ≥ 0.80.

**Secondary (prompted AUROC cells, NLA and Oracle Lens, 16 cells):** reported with the same rule, kept separate
because the prompted score is a forced choice between two phrasings. The Oracle Lens prompted cell is the one that
generalised in exp 18 (9 of 10) and is reported for continuity; it does not enter the primary accuracy.

Both scorecards carry a Wilson 95% CI and the per-reader confusion table. The two revised NLA rules are also
reported on their own (16 cells) so that a pass or fail of the revision is visible apart from the carried-over
rules.

## 6. Bank and instruments (filled 27 Oct 13:15 IST, before any reader ran)

Bank `experiments/19_natural_bank/results/items_base.json`, sha256 `7ddfbceac1a5510156b1037239606536e50484a61f93c057629c26fa41525460`, 1,600 items; lexical cue per row in
`experiments/19_natural_bank/RESULTS.md` (json_depth 0.94 and html_depth 1.00 are over the 0.65 admission line, so
their J-lens cells are not scored; the other six rows are 0.45 to 0.56).

Instruments (`results/instruments19.json`, sha256 `aab6317039a6568e7534a468afbc03c72eec47eb1fd05615bdb031e2a45c0183`) and the predicted cells
(`results/predictions19.md`, sha256 `c8b593c0aabde7ba7d0c6eddf4601fd950687738918db1df9b37c39d8ae242ea`):

Thresholds: readable ≥ 0.8, span ≥ 0.8, statement ≥ 0.3, used z ≥ 3.0, J-enriched ≥ 1.5. Predicted cells: H = predicted high, L = predicted low.

| row | next-token set | constrains next | short value | readable | span (frozen probe) | stmt | used z | J-enr | NLA prompted | NLA states | NLA names complement | OL prompted | OL exhibits | OL states | J-lens prompted |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| py_string | '"' "'" | yes | yes | 1.00* | 0.85* | 0.04 | 6.56* | 1.17 | H | L | L | H | H | L | L |
| js_string | '"' "'" | yes | yes | 0.99* | 0.82* | 0.01 | 0.70 | 1.22 | H | L | L | H | H | L | L |
| latex_math | '$' | yes | yes | 0.99* | 0.83* | 0.05 | 25.55* | 1.31 | H | L | L | H | H | L | L |
| py_arg | ',' ')' | yes | yes | 0.99* | 0.89* | 0.23 | 3.16* | 1.10 | H | L | H | H | H | L | L |
| json_depth | '}' | yes | yes | 0.99* | 0.87* | 0.14 | -1.98 | 1.29 | H | L | H | H | H | L | L |
| html_depth | '</' | yes | yes | 0.99* | 0.74 | 0.04 | -4.62* | 1.32 | H | L | H | L | L | L | L |
| md_heading | none | no | yes | 0.63 | 0.51 | 0.03 | n/a | 1.48 | L | L | H | L | L | L | L |
| md_list | none | no | yes | 0.65 | 0.64 | 0.07 | n/a | 1.42 | L | L | H | L | L | L | L |

* = passes its threshold.

Reading: all six code and markup rows are readable and span-sayable except html_depth (span 0.74); the two
Markdown rows are not readable at layer 42 on real text (0.63, 0.65) where their synthetic versions in exp 18 were
(1.00); no row is statement-sayable (max 0.23), so the revised rule predicts the NLA states nothing on this bank and
names the complement on the five both-positive rows only; no row is J-enriched.

## 7. Budget

One GPU at a time; about $10 GPU (instruments, generation, scoring at the exp 18 rates), about $3 judge and mining.

## 8. Public-source rebuild (exp 22, 27 Oct): same rows, same rules, second prospective test

Exp 19's bank drew Markdown and HTML items from local files of unverified provenance and is not published. Exp 22 rebuilds
the same eight rows from published sources only (package distributions on this machine: site-packages, dist-packages,
node_modules, the bun and uv caches, conda pkgs; arXiv e-prints for LaTeX; arXiv HTML renderings for HTML), each item
carrying its source class, with a new seed (2222) and the same generators, lexical-cue check (0.65), instruments,
thresholds, prediction rules (§3), scoring split (§5) and cell thresholds. Predictions are filled and hashed below before
any reader runs. The result is a second prospective test of the revised NLA rules and of the carried-over rules, on a
bank that will be tracked in the repository.

The HTML row is dropped from exp 22: the public HTML on this machine (176 package files, and arXiv's HTML renderings, which wrap all text deeply) yields 36 source pages with both shallow and deep text, 88 items. Eight rows remain: the seven carried from exp 19 and the positive control (code_lang, Python vs JavaScript, both values positive states, no next-token set).

Bank and instruments (filled 27 Oct 16:30 IST, before any reader ran). Bank sha256 `3260ea2697b33d7516190c3c77027d8dd9275e6348c2aadde0c41a2fa4d5aa08`, 1,600 items, every
item from a package directory or arXiv (`results/build_info.json` has the per-row source counts). Lexical cue: py_string
0.58, js_string 0.51, latex_math 0.48, py_arg 0.51, md_heading 0.63, md_list 0.54 pass; json_depth 0.96 and code_lang
0.98 are cued and their J-lens cells are not scored. Instruments `results/instruments22.json` sha256 `ab6b2f787ae7064ce1cef8c9d07956dbf0b7e5456689887fecea08d4f209d2bf`;
predictions `results/predictions22.md` sha256 `c0fe07c91bc67cf48d2af1721bb577b67627b619208e138d445073e9431bd6c6`:

Thresholds: readable ≥ 0.8, span ≥ 0.8, statement ≥ 0.3, used z ≥ 3.0, J-enriched ≥ 1.5. Predicted cells: H = predicted high, L = predicted low.

| row | next-token set | constrains next | short value | readable | span (frozen probe) | stmt | used z | J-enr | NLA prompted | NLA states | NLA names complement | OL prompted | OL exhibits | OL states | J-lens prompted |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| py_string | '"' "'" | yes | yes | 1.00* | 0.84* | -0.02 | 10.72* | 1.06 | H | L | L | H | H | L | L |
| js_string | '"' "'" | yes | yes | 0.98* | 0.77 | 0.01 | -1.97 | 1.16 | H | L | L | L | L | L | L |
| latex_math | '$' | yes | yes | 1.00* | 0.85* | 0.02 | 24.66* | 1.20 | H | L | L | H | H | L | L |
| py_arg | ',' ')' | yes | yes | 0.95* | 0.88* | 0.21 | -1.24 | 1.06 | H | L | H | H | H | L | L |
| json_depth | '}' | yes | yes | 1.00* | 0.92* | 0.13 | -1.75 | 1.33 | H | L | H | H | H | L | L |
| md_heading | none | no | yes | 0.72 | 0.62 | 0.06 | n/a | 1.40 | L | L | H | L | L | L | L |
| md_list | none | no | yes | 0.49 | 0.50 | -0.03 | n/a | 1.04 | L | L | H | L | L | L | L |
| code_lang | none | no | yes | 0.99* | 0.99* | 0.30 | n/a | 1.59* | H | L | H | H | H | L | H |

* = passes its threshold.

Note recorded before the readers run: the positive control's statement shift is 0.2976, just under the 0.30 line,
so the fixed rule predicts the NLA does NOT state the code language. We expect this prediction to fail (the NLA names
the language of code freely in every mining readout we have read); if it does, the failure is the rule's threshold or
the statement instrument, and it is reported as such, not re-thresholded.
