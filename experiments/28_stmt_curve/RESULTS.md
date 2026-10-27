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

## Results

(placeholder: the run has not been launched)
