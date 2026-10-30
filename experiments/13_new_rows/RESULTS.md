# 13 New concept rows for the per-reader recall table (PILOT / EXPLORATORY)

**Why.** Neel's 11 Oct thread asked, per structural concept type, how often each reader finds it, at one
layer. Exp 12 answered with the 14 exp 07 rows. This adds five position-in-text rows (the kind Neel called
interesting), two formatting rows (Camila's kind, for comparison) and a fresh line-position bank, built and
scored exactly as exp 07/12: generated items with the true value known, NLA / OLens / J-lens on the same
L42 activation, prompted AUROC on 160 held-out items and judged spontaneous recall on 40 items × 8 samples.

Date 2026-10-22. Modal app `tq7` (stages s0–s4), two H100/H200 containers at most. Readouts: OLens free
text at L42 only (exp 12 showed 40/42/44 interchangeable); activations captured at 40/42/44 for prompted
scores. Judge: Gemini 3.8 Flash, 224 calls, $1.02; extraction 14 calls, $0.15.

| Row | Values | Group |
|---|---|---|
| sentences into paragraph | 1st–2nd vs 5th+ sentence (6–15 words into it in both) | position-in-text |
| table column | first cell vs 4th–6th cell of a markdown table row | position-in-text |
| line in function body | body line 2–3 vs 9–12, flat indentation | position-in-text |
| inside a parenthesis | 3+ words into an open "(" vs 4+ words after ")" | position-in-text |
| words left under a stated limit | "Write exactly N words": 20–25 left vs 2–5 left | position-in-text |
| Q&A vs prose layout | same passage as "Question: … Answer: …" vs plain | formatting |
| list nesting depth | top-level vs third-level markdown bullet | formatting |
| line position, fresh items | exp 07's generator on unseen paragraphs; frozen nj1_col templates only | position-in-text |

Code: `feature_spec.py`, `items_build.py` (+ `test_items_build.py`, 9 tests), `modal13.py`, exp 07's
`extract_templates.py` / `select_freeze.py` (copied, feature tables patched), exp 12's judge via env vars,
`combine13.py`. Outputs in `results/`: `items_base.json`, `gen_*`, `candidates.json`, `score_*`,
`selection_table.json`, `templates_frozen.json`, `judged/`, `recall_table.md`, `combined_all.{md,json,png}`.

## Spontaneous recall (judged free readouts, L42)

| Row | NLA states it | NLA any-of-8 | OLens states it |
|---|---|---|---|
| table column | 41% (118/160 "first" items, 12/160 "late") | 57% | 0% |
| inside a parenthesis | 30% (97/160 inside items; never says "outside") | 38% | 0% |
| list nesting depth | 4% | 18% | 0% |
| Q&A vs prose | 4% | 12% | 0% |
| sentences into paragraph | 0% | 0% | 0% |
| line in function body | 0% | 0% | 0% |
| words left under limit | 0% | 0% | 0% |

- **OLens states nothing on any new row** (0 of 2,240), as on the 14 earlier rows (1 of 4,480 there).
- **The NLA's new statements are one-sided and about pending structure.** It says "a new row is beginning" /
  "first column" on first-cell items and almost never names a late column; it says "an open parenthetical"
  on inside items and never "outside a parenthesis". Stated values are correct whenever stated. This is the
  same habit seen on bullet index and quote state in exp 12: the NLA reports structure that constrains what
  comes next, not structure as a property of the text.
- **Three more position-in-text counts are never spoken:** sentences into the paragraph, line number in a code
  body, words remaining under an explicit limit. Together with line position, sentence position and
  indentation from exp 12, that is six count-like position-in-text concepts with 0% spontaneous mention.
- **Formatting rows (Camila's category) are low**, 4% stated for both, so "the NLA's unique content is
  formatting" is not what we see on controlled items; what it states is completion-relevant structure.

Spot-checks (NLA, judged "stated"): table column, true first: "Table structure continuation: pipe-delimited
rows …" with the row start named; parenthesis, true inside: "Tuvalu's athlete Barbara Hepu is being described
with a parenthetical clarifying her sport"; list depth, true shallow: "A second top-level section is beginning".
Judge caveat: on table column the judge accepts "a new row/entry is beginning" as stating the first column.

## Prompted scores (AUROC, 160 held-out items; mean of the 3 selected templates, L42)

| Row | NLA | OLens | J-lens | length alone | NLA / OLens on length-overlap items (n) |
|---|---|---|---|---|---|
| table column | 1.00 | 0.74 | 0.64 (fixed) | 0.32 | 1.00 / 0.67 (133) |
| inside a parenthesis | 0.96 | 0.73 | 0.47 | 0.29 | 0.98 / 0.76 (143) |
| sentences into paragraph | 0.99 | 0.65 | 0.42 | 0.00 (perfectly separable) | no overlap (n = 2) |
| line in function body | 0.99 | 0.71 | 0.48 | 0.00 (perfectly separable) | no overlap (n = 0) |
| list nesting depth | 0.96 | 0.88 | 0.96 (fixed) | 0.42 | 1.00 / 1.00 (82; length alone 0.83 there) |
| Q&A vs prose | 0.79 (fixed) | 0.56 | 0.54 | 0.86 | 0.63 / 0.54 (119; length alone 0.75) |
| words left under limit | 0.58 | 0.56 | 0.55 | 0.08 | 0.45 / 0.46 (85) |
| line position, fresh items | 0.51 (fixed; mean of 3 0.48) | 0.49 | 0.53 (fixed) | – | – |

- **Two clean "NLA reads it, OLens half, J-lens blind" cells, robust to length:** table column (1.00 vs
  0.67 vs 0.64 on length-matched items) and parenthesis state (0.98 vs 0.76 vs 0.47). Both are also the two
  rows the NLA states unprompted (41% / 30%). These are the first prompted cells in the whole table where the
  NLA clearly beats OLens on a position-in-text concept; on the 14 earlier rows OLens ≥ NLA on 7.
- **Sentence count and code-line number are confounded with prefix length by construction** (late items are
  always longer; no overlap to re-score). The 0.99s say the NLA separates them but not that it does so by
  counting sentences or lines rather than tokens. Redesign for the next pass: pad early items with an
  unrelated preamble so token length is matched.
- **List depth** is read by all three readers (J-lens 0.96 on the fixed words "top / nested"); length is also
  predictive there (0.83 in the overlap), so it is not a clean cell.
- **Q&A layout** is weak for every reader once length is accounted for; **words-left** is at chance for all.
- **Fresh line position: the pilot's 0.60 did not replicate** (NLA 0.51 on the fixed label and 0.48 on the mean of three, OLens 0.49, J-lens 0.53 with the frozen
  nj1_col templates on 160 unseen items). This is the pre-registered P1 cell run informally on fresh items:
  the predicted branch was "P1 fails", and it does. Line position is at chance for all three readers.

`length_control.py` → `results/length_control.json`.

## Caveats

- Sentence count and code-line rows are fully length-confounded (see prompted section); table column and
  parenthesis are not (length AUROC 0.32 / 0.29, readers unchanged on length-matched items).
- 40 × 8 samples per spontaneous cell; one judge (Gemini), hand-checked on three rows.
- The fresh col bank is scored with exp 07's frozen templates and has no spontaneous column.
