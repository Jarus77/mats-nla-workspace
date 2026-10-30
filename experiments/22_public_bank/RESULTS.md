# Exp 22: the natural-text prediction test rebuilt from public sources only (second prospective test)

Pre-registration: `notes/prereg_natural_bank.md` §8 (rules and thresholds of §1–§5 carried over unchanged; bank,
instruments and predictions hashed at 16:30 IST on 27 Oct before any reader ran). Pipeline: exp 19's, cloned
(`modal22.py`, app `p8sm`). The bank is tracked in this repository (`results/items_base.json`, sha256
`3260ea2697b33d7516190c3c77027d8dd9275e6348c2aadde0c41a2fa4d5aa08`): every item comes from a package directory on the
build machine (site-packages, dist-packages, node_modules, the bun and uv caches, conda pkgs) or from an arXiv e-print,
and `meta.src` records which. Seed 2222. The HTML row of exp 19 is dropped (public HTML on the machine yields 36
usable pages); a positive-control row is added.

## Bank (27 Oct)

Eight rows, 200 read points each (20 + 20 mining, 80 + 80 selection), both values of a row from the same source files,
clustered by file.

| row | values | source | lexical cue (word tokens, last 48) | J-lens cell scored |
|---|---|---|---:|---|
| py_string | inside / outside a string literal | packages (.py) | 0.58 | yes |
| js_string | inside / outside a string literal | packages (.js) | 0.51 | yes |
| latex_math | inside / outside `$...$` | arXiv (.tex) | 0.48 | yes |
| py_arg | first / third call argument | packages (.py) | 0.51 | yes |
| json_depth | deep / shallow key | packages (.json) | 0.96 | no (cued) |
| md_heading | heading level 1 / 3 | packages (.md) | 0.63 | yes |
| md_list | top-level / nested list item | packages (.md) | 0.54 | yes |
| code_lang | Python / JavaScript, outside strings (positive control) | packages | 0.98 | no (cued) |

## Instruments (before any reader ran; `results/predictions22.md`, hashes in the pre-registration §8)

| row | readable | span (frozen probe) | statement | used z | J-enr |
|---|---:|---:|---:|---:|---:|
| py_string | 1.00 | 0.84 | -0.02 | +10.7 | 1.06 |
| js_string | 0.98 | 0.77 | 0.01 | -2.0 | 1.16 |
| latex_math | 1.00 | 0.85 | 0.02 | +24.7 | 1.20 |
| py_arg | 0.95 | 0.88 | 0.21 | -1.2 | 1.06 |
| json_depth | 1.00 | 0.92 | 0.13 | -1.8 | 1.33 |
| md_heading | 0.72 | 0.62 | 0.06 | n/a | 1.40 |
| md_list | 0.49 | 0.50 | -0.03 | n/a | 1.04 |
| code_lang | 0.99 | 0.99 | 0.30 (0.2976) | n/a | 1.59 |

The exp 19 picture again on a public bank: every structural row is linearly readable, the Markdown rows are not
(0.72, 0.49), no row is statement-sayable. The positive control sits at 0.2976, under the 0.30 line, so the fixed rule
predicts "NLA does not state the language"; the pre-registration records, before the readers ran, that we expected this
cell to fail.

## Scorecard (`results/scorecard22.md`, `results/score22.json`; readers at layer 42, Oracle Lens also at 36/44/52)

**Primary scorecard (free-readout cells, J-lens on the six uncued rows): 30 of 38, 0.79 [0.64, 0.89].** Without the
Oracle Lens "exhibits" cell (judge-scored; see below): 26 of 30, 0.87 [0.70, 0.95]. **Secondary (prompted forced-choice
cells): 13 of 16, 0.81 [0.57, 0.93].** All 56 cells, the exp 18 convention: 44 of 56, 0.79 [0.66, 0.87], against the
declared 0.80. Without the positive-control row: primary 27 of 34 (0.79), secondary 11 of 14 (0.79).

| rule | score | what happened |
|---|---|---|
| NLA states iff statement-sayable (revised) | 6 / 8 | no natural row stated (max 0.14 on the seven); misses: json_depth 0.25 (predicted low at stmt 0.13; exp 19 had 0.19 on the same row) and the positive control, stated at 0.70 as expected (stmt 0.2976) |
| NLA names the complement iff both values are positive states (revised) | 6 / 8 | named on py_arg 0.71, json 0.68, heading 1.00, code_lang 0.78; the same two misses as exp 19: latex_math 0.40 (an absence row, named) and md_list 0.00 (both-positive, not named; the row is not readable) |
| Oracle Lens never states | 8 / 8 | max 1% |
| Oracle Lens exhibits iff span-sayable | 4 / 8 | judge 0.00 on every natural row for the third experiment running; the judge does see exhibition on the positive control (0.78, correctly predicted high) |
| J-lens high iff J-enriched or statement-sayable | 6 / 6 | all predicted low, all low (0.51 to 0.66); the cued rows excluded as declared (json 0.79, code_lang 0.93, the latter correctly predicted high) |
| NLA prompted high iff readable (secondary) | 6 / 8 | misses latex_math 0.82 and py_arg 0.65 under the 0.85 line; both unreadable rows low as predicted |
| Oracle Lens prompted high iff span-sayable (secondary) | 7 / 8 | one miss in the opposite direction from exp 19: js_string predicted low (span 0.77) and detected at 0.87; py_string 0.91 and json_depth 0.88 are detected here (0.52 and 0.64 in exp 19) |

Confusion, predicted x observed: NLA HH 8, HL 3, LH 3, LL 10; Oracle Lens HH 6, HL 4, LH 1, LL 13; J-lens (all eight
rows) HH 1, LH 1, LL 6.

## Reading

1. **The exp 19 result replicates on a public bank built from different files with a different seed.** The two
   banks share seven rows (49 cells); on those the two scorecards agree hit for miss on 42 of 49 cells (observed
   high/low on 44 of 49), and the four revised-rule misses include the same two complement-naming misses (latex_math
   named at 0.40 and 0.41; md_list unnamed on both). The revised NLA rules score 12 of 16 here against 14 of 16 in
   exp 19; the two new misses are json_depth creeping over the 0.20 observed line (0.25) and the positive control.
   Beside an always-"low" predictor on the same cells the rules do not win on the primary scorecard: exp 19 primary
   31 of 38 against 33 of 38 for always-low, exp 22 30 of 38 against 30 of 38; all cells, exp 19 44 of 56 against
   45 of 56, exp 22 44 of 56 against 36 of 56 (exp 18: 44 of 70 against 40 of 70). On the primary scorecard the rules
   tie or lose to always-low. Their margin over the baseline among the NLA cells comes from the complement-naming rule:
   the two revised NLA rules score 14 of 16 against 11 of 16 for always-low in exp 19 and 12 of 16 against 9 of 16
   here.
2. **The positive control failed as pre-registered, and that is informative about the instrument, not the thesis.**
   Python-vs-JavaScript is the one row where the NLA states the property freely (0.70), the judge sees Oracle Lens
   exhibiting it (0.78), and the J-lens reads it (0.93). Its statement shift is 0.2976. Every natural structural row
   sits at or under 0.21 and is stated at or under 0.25. The statement instrument orders the rows correctly; the 0.30
   line is a hair too high for a feature that is stated 70% of the time. The line is not moved; the ordering is the
   finding.
3. **Oracle Lens never states a property** (8 of 8, max 1%), flat across layers 36 to 52 on every row, as in exps 12,
   13, 18 and 19.
4. **The prompted Oracle Lens cell is phrasing-bound, in both directions.** On py_string the fixed phrasing is reversed
   again (0.31 at layer 42, as 0.31 in exp 19) while a mined phrasing reaches 0.91; on json_depth fixed 0.35, mined
   0.88. In exp 19 no mined phrasing reached the line on those rows and the cell missed; here one did. js_string is
   detected at 0.87 with span 0.77, the only LH cell. Forced-choice detection depends on finding a phrasing the lens
   answers, which is what Neel's "off-distribution" remark predicts; the free-readout cells do not.
5. **The judge cannot score exhibition of structure but can score exhibition of language.** 0.00 on all seven natural
   rows (24 rows across exps 18, 19 and 22) and 0.78 on code_lang. The judge is not blind; the readers' free text
   exhibits a code language in a way a reader of the text recognises and exhibits structural state in a way only a
   classifier on the text recognises (exp 20's reconstructor and text-bag measures). The reconstructor-based measure
   was not run on this bank.
6. **The J-lens cells are uninformative on structural rows** (all low, none J-enriched), and the one J-enriched row
   (code_lang, 1.59) is the one the J-lens reads (0.93), as the rule says.

## What differs from exp 19

| | exp 19 | exp 22 |
|---|---|---|
| bank | local files, unpublished | package directories and arXiv, tracked |
| rows | 8 incl. html_depth | 8 incl. code_lang positive control |
| primary | 31 / 38 (0.82) | 30 / 38 (0.79) |
| primary, always-low baseline | 33 / 38 | 30 / 38 |
| primary without the judged exhibits cell | 28 / 30 (0.93) | 26 / 30 (0.87) |
| secondary | 11 / 16 (0.69) | 13 / 16 (0.81) |
| revised NLA rules | 14 / 16 | 12 / 16 (one miss the pre-registered positive control) |
| OL never states | 8 / 8 | 8 / 8 |
| OL prompted misses | py_string, json_depth (HL) | js_string (LH) |

Cost: about $9 GPU (instruments one H100-hour and one H200-hour, readouts and scoring about the same), $1.50 judge
and mining. Readouts (`results/gen_*.json`) are not tracked (size); per-item scores, judge verdicts, the frozen
templates and the scorecard are.
