# Exp 28: the statement shift as a graded instrument, on every row that has a stated rate

**Question.** The theory review (`notes/draft_critique_theory.md`, 28 Oct) assembled every (row, statement shift, NLA
stated rate) pair in the record, 30 of them, and found that the rule "the NLA states c iff `stmt` >= 0.30" is carried
by three language rows: "Over 30 (row, stmt, stated) pairs the Spearman is 0.43 (p = 0.01); drop the three language
rows and it is 0.21 (p = 0.15), Pearson 0.10. The scatter is two clusters, not a step on a continuum." Four rows the
derivation did not count are misses for the rule (table column 0.07 -> stated 0.41, parenthesis -0.05 -> 0.30, exp 18
call argument 0.09 -> 0.33, bullet index 0.23 -> 0.25), and the shift itself "is one statement pair per row, two
reconstructor passes, no error bar; exp 14 shows phrasing moves it by 0.34 on one row, wider than the whole negative
cluster." Seventeen exp 12/13 concepts have stated rates but no shift at all. So: measured with eight paraphrase
pairs per row, on every row that has a stated rate, is the statement shift a graded predictor of the stated rate, or
a type label wearing a continuous coat?

**Rows.** The union of exps 12 (exp 07's 14 concepts, judged in exp 12), 13 (seven judged rows), 18 (ten), 19 (eight)
and 22 (eight): 47 rows, every one with an NLA stated rate over 320 readouts (40 items x 8) and a frozen NLA statement
pair in its source's `FIXED["nla"]`. `lib28.collect_rows()` reads the specs and scorecards and writes
`results/rows28.json`; nothing with a stated rate was left out (`results/rows28_missing.json` is empty; exp 13's
`nj1_col_fresh` has no judged readouts and so no stated rate). Concept type is declared in `lib28.SELF_INDUCING`
before the run: self-inducing = a whole-text property a short text can itself exhibit, the four concepts exp 12's
judge found exhibited (language, code language, caps, tense) plus exp 18's language row and exp 22's code-language
row, six rows; the other 41 (position in text, nesting, upcoming-word syntax, the stance word) are structural. The
memo's four counter-rows and its language rows (plus exp 12's `bk_code`) are flagged in the records.

**Pipeline** (`run28.sh`). (1) `paraphrase28.py` (laptop, Gemini 3.8 Flash via OpenRouter, exp 06's `Client`, one
call per row): for each row's fixed pair, eight paraphrase pairs in eight fixed registers (`lib28.REGISTERS`:
declarative, "the text is", "we are currently", short, long, a reviewer's note, the reader's point of view, passive or
nominal), the two values aligned and each pair identical except for the value words; validated by `lib28.parse_pairs`,
frozen in `results/paraphrases28.json` with its sha256 in `results/paraphrases28.sha256` before the GPU stage runs.
(2) `modal28.py` stage `stmt` (app `y4hb`, function `q8nr`, H200, NLA critic `load_nla(load_ar=True)`): per row,
the selection items and layer-42 activations from the volume path in `rows28.json` (exp 22 `acts22_{row}.pt`, exp 19
`acts19_{row}.pt`, exp 18 `acts18_{row}.pt`, exps 12/13 `07_template_mining/acts_{row}.pt` and `13_new_rows/acts_{row}.pt`
with the items in `gen_ol_{row}.json`), the true class direction d (marked minus unmarked mean) and gap, and for the
fixed pair and the eight paraphrase pairs the shift ((rec(A) - rec(B)) . d_hat) / gap, exactly modal22's `stmt`;
also `span` (exp 14's frozen-probe AUROC on the critic's rebuild of the true last-32-token span, `lib22.frozen_probe_auroc`),
`span_gap` and `readable` (grouped 5-fold probe) for every row, so the exp 12/13 rows get them and the others get a
consistency check against their source. One file per row, `stmt28_{experiment}_{row}.json`, committed after each
row; resumable. (3) `analyze28.py`: `results/table28.md`, `results/fig28.png` (x = paraphrase-mean shift with
min-max whiskers, y = stated rate, colour = type, the 0.30 line dashed, labels on the counter-rows and language
rows), `results/stats28.json` (Spearman and Pearson with a 20,000-shuffle permutation p over all rows, within each
type and without the language rows; the memo's band table; the AUROC of the shift for stated >= 0.30; binomial
logistic fits of the stated rate on the shift, on the type indicator and on both, with AIC; `lib28.logistic_fit`, an
IRLS, since the venv has neither statsmodels nor sklearn). Tests: `test_lib28.py` (16: the formula against exp 14's
`ar_lib.statement_shift` and modal22's form on a hand case and random data, the loaders on synthetic blobs, the
permutation correlations, the logistic fit on a planted slope, the band table, the paraphrase parser, `collect_rows`
on a synthetic repo and on the real one).

**Cost.** Paraphrases: 47 calls of about 700 input and 500 output tokens, under $1. GPU: one H200 container, per row
one critic pass over 18 statements and one over 160 spans (exp 22's instr2 did the same per row in about a minute),
47 rows, about one hour with loading: about $8. Analysis is CPU. About $10 in all.

## Predictions written before the run

1. **Paraphrase ranges are wide.** The min-to-max range of the eight paraphrase shifts exceeds 0.15 on at least half
   the rows (24 of 47 or more), so a single-pair shift is not a precise number and the memo's "0.2976 against 0.30"
   is a difference inside the phrasing noise.
2. **The four counter-rows stay counter-rows.** Table column (13), parenthesis (13), call argument (18) and bullet
   index (12) keep paraphrase-mean shifts under 0.20 while stated at 0.25 to 0.41, so they remain counterexamples to
   the 0.30 rule as stated, and the rule has to be re-worded as a two-type account (self-inducing rows stated,
   structural rows mostly not) plus a warm-start register term for the pending-structure statements.
3. **Correlation lives between the types, not within.** Spearman over all 47 rows between 0.3 and 0.6; within the
   structural rows alone under 0.3 (no graded relation inside the lower cluster). Within the six self-inducing rows the
   number is reported but, at n = 6, not read.
4. **The shift does not beat the type label.** The binomial logistic of the stated rate on the paraphrase-mean shift
   does not beat the type-indicator-only fit by more than 4 AIC (AIC(type) - AIC(shift) > -4).

What would change the account: if (3) fails upward (Spearman within the structural rows >= 0.4 with p < 0.05) the
shift does graded work inside the lower cluster and C2 survives as a curve; if (4) fails (the shift beats the type
indicator by more than 4 AIC) the continuous instrument carries information the label does not. Tense (12,
`nj2_tense`, stated 0.00) is the live self-inducing row: a paraphrase-mean shift at or above 0.4 there is the memo's
F2 (the reward pays and the NLA declines).

## Results (27 Oct, 23:53 UTC; `results/table28.md`, `results/stats28.json`, `results/fig28.png`; 47 rows, eight paraphrase pairs each, one H200 pass of 9 minutes plus 47 Gemini calls, about $3)

**The scatter.** Paraphrase-mean statement shift against the NLA's stated rate, 47 rows from exps 12, 13, 18, 19, 22:

| band of the shift | rows | mean stated | range |
|---|---:|---:|---|
| under 0.10 | 32 | 0.09 | 0.00 to 0.41 (table column 0.41, call argument 0.33, parenthesis 0.30 in this band) |
| 0.10 to 0.25 | 13 | 0.22 | 0.00 to 0.82 (code language, exp 12, at 0.16 shift stated 0.82; tense at 0.15 stated 0.00) |
| 0.25 and above | 2 | 0.88 | the two natural-language rows, 0.36 and 0.49 |

Correlation of the shift with the stated rate: Spearman 0.40 (p = 0.006) over all rows; 0.23 (p = 0.14) without
the four language rows; 0.28 (p = 0.08) within the 41 structural rows. AUROC of the shift for "stated at or above
0.30": 0.76. Paraphrase ranges are narrow on most structural rows (SD 0.01 to 0.08; median range 0.10) and wider on
the natural-language rows (min to max 0.24 to 0.35), though five structural rows also exceed 0.15 (up to 0.24 on
sentence count). The fixed-pair shifts recomputed here match the source experiments' values within 0.03
on every row, and exp 14's four values within 0.10.

Logistic fits of the stated rate (binomial, 320 readouts per row): shift alone beats the type indicator alone by
314 AIC; shift plus type beats shift alone by 176 AIC; so the shift carries information beyond the type label and
the type label carries information beyond the shift.

### Scoring the predictions

| prediction | result |
|---|---|
| 1. paraphrase range above 0.15 on at least half the rows | fails: 8 of 47 rows (`stats28.json`): the two natural-language rows (0.35, 0.24) and exp 12's code language (0.17), and five structural rows (sentence count 0.24, quote 0.22, call argument in exps 22 and 19 0.19 and 0.17, predicate position 0.16); tense is 0.10; median range 0.10, so a single pair was a fair measurement on most structural rows |
| 2. the four counter-rows keep paraphrase-mean shifts under 0.20 | holds: table column 0.05, call argument 0.05, parenthesis 0.00, bullet index 0.14 |
| 3. Spearman 0.3 to 0.6 overall, under 0.3 within structural rows | holds: 0.40 and 0.28 |
| 4. a logistic in the shift does not beat the type indicator by more than 4 AIC | fails: it beats it by 314, because the shift also separates the language rows from the rest more finely than a binary type does |

## Reading

1. **The 0.30 rule is dead as a threshold.** Seven rows are stated at 0.25 or more with shifts under 0.17 (table
   column 0.41, call argument 0.33, predicate position 0.31, parenthesis 0.30, number agreement 0.28, bullet index
   0.25, JSON depth 0.25), and exp 12's code-language row is stated 0.82 at a shift of 0.16. Any line that admits
   the language rows admits none of these, and any line that admits these admits most of the unstated rows.
2. **What the shift does measure.** It separates the natural-language rows (0.36, 0.49) from everything else, with
   the code-language rows next (0.16, 0.24), and it tracks the stated rate across the whole set at Spearman 0.40.
   Inside the structural rows the relation is weak and not significant. So the shift is a graded measure of how
   far a statement re-induces the state in the reconstructor, and that quantity governs naming only at the top of
   its range.
3. **The NLA's structural naming is not predicted by the reconstructor.** The rows it names most (table column,
   call argument, parenthesis, predicate position) have shifts indistinguishable from the rows it never names
   (LaTeX math, HTML depth, string state). What they share is being pending structure a describing register
   mentions ("a new row is beginning", "the third argument"), which is a property of the warm start, not of the
   reward. Exp 29 tests that directly along the training trajectory; exp 30 shows the naming can be added by
   supervision without touching the reward's score.
4. **Consequence C2 of the derivation is re-worded**: a statement is rewarded only through the shift, so the
   reward never pays for structural statements (true for every structural row: shifts 0.00 to 0.19), and whether
   the NLA makes them anyway is decided by its register. The sentence "the NLA states a property iff the statement
   shift clears 0.30" is withdrawn.
