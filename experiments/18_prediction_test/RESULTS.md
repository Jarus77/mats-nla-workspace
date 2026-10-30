# 18 — Prediction test: does the objective account predict the readers on concepts it has never seen? (26 Oct 2026)

**Verdict against the pre-registered bar: no, not as written.** Accuracy 0.63 [0.51, 0.73] over 70 scored cells
(bar 0.80; an always-"low" predictor scores 40 of 70 on the same cells against the rules' 44). The Oracle Lens half of the account is predictive (prompted 9/10, never-states 10/10); the NLA
"states iff used" rule (5/10), the blanket "never names the complement" rule (4/10) and the J-lens rule (2/10)
are not. Pre-registration: `notes/prereg_prediction_test.md` (instruments and predictions hashed before any reader
ran; two clarifications disclosed there). Scorecard: `results/scorecard18.md`; everything: `results/score18.json`.

## Design

Ten rows, 200 items each (20 + 20 mining, 80 + 80 selection), length-matched: JSON depth, LaTeX math, Python
string, call argument, Markdown heading, CSV field, HTML depth, stance word, and language and quote controls
(`items_build18.py`, bank sha256 `44308d…cd36`). Instruments without a reader (D41): readable (probe), sayable
by span (frozen probe on the NLA reconstructor's rebuild of the true 32-token span), sayable by statement, used
(centred class-mean push at r11 vs 20 isotropic nulls), J-enrichment. Predictions per reader cell written and
hashed (`results/predictions18.md`, sha256 `1edd82…8aad`; `2db980…77a6` is `results/instruments18.json`). Readers: free readouts k = 8 on the mining split
(NLA, OLens@42) judged by Gemini 3.8 Flash for stated / exhibited; phrasings mined from those readouts and
selected on the selection split with exp 07's procedure (`templates_frozen.json`); prompted log-odds AUROC
(max of the 3-template mean and the fixed label) for NLA@42, OLens@36/42/44/52, J-lens@42; the NLA's
complement-naming rate from the fixed pair's sign on unmarked items. Cost: about $8 GPU, $2.5 judge and mining.

## Scorecard (H / L = predicted high / low; number = observed; ok / MISS against the cell's declared threshold)

| row | NLA prompted ≥0.85 | NLA states ≥0.20 | NLA names complement ≥0.20 | OL prompted ≥0.70 | OL exhibits ≥0.30 | OL states ≥0.05 | J-lens prompted ≥0.70 |
|---|---|---|---|---|---|---|---|
| json_depth | H / 1.00 ok | L / 0.25 MISS | L / 0.85 MISS | H / 0.99 ok | H / 0.00 MISS | L / 0.03 ok | L / 0.87 MISS |
| latex_math | H / 0.95 ok | H / 0.00 MISS | L / 0.00 ok | H / 1.00 ok | H / 0.00 MISS | L / 0.00 ok | L / 1.00 MISS |
| py_string | H / 1.00 ok | H / 0.03 MISS | L / 0.00 ok | H / 1.00 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.95 MISS |
| call_arg | H / 1.00 ok | H / 0.33 ok | L / 1.00 MISS | H / 1.00 ok | H / 0.00 MISS | L / 0.00 ok | L / 0.99 MISS |
| md_heading | H / 0.92 ok | L / 0.00 ok | L / 0.96 MISS | L / 0.65 ok | L / 0.00 ok | L / 0.00 ok | L / 0.74 MISS |
| csv_col | H / 0.96 ok | L / 0.13 ok | L / 1.00 MISS | L / 0.89 MISS | L / 0.00 ok | L / 0.00 ok | L / 0.84 MISS |
| html_depth | H / 0.84 MISS | L / 0.00 ok | L / 0.21 MISS | L / 0.57 ok | L / 0.00 ok | L / 0.00 ok | L / 0.58 ok |
| stance_word | L / 0.55 ok | L / 0.00 ok | L / 0.00 ok | L / 0.59 ok | L / 0.00 ok | L / 0.00 ok | H / 0.52 MISS |
| ij2_lang | H / 1.00 ok | L / 0.95 MISS | L / 0.97 MISS | H / 1.00 ok | H / 0.72 ok | L / 0.00 ok | H / 0.98 ok |
| bk_quote | H / 0.79 MISS | H / 0.08 MISS | L / 0.04 ok | L / 0.62 ok | L / 0.00 ok | L / 0.00 ok | H / 0.55 MISS |

Confusion (predicted × observed): NLA HH 8, HL 5, LH 8, LL 9; OLens HH 6, HL 4, LH 1, LL 19; J-lens HH 1, HL 2, LH 6, LL 1.

## What held

1. **Oracle Lens prompted: 9 of 10.** The span-sayability rule put the lens high on JSON, LaTeX, string,
   argument and language (observed 0.99 to 1.00) and low on heading, HTML, quote and stance (0.57 to 0.65). The
   decisive cells were the three rows where readable says yes and span-sayable says no (heading, HTML, CSV): two
   of three came out as predicted. CSV field is the miss (0.89 at a span score of 0.71). **Flat across layers**:
   every row is within 0.08 across OLens@36/42/44/52 (heading 0.59 to 0.65, HTML 0.48 to 0.57, quote 0.61 to 0.63),
   so the blind spot is not a layer accident, as in exp 12.
2. **Oracle Lens never states: 10 of 10** (maximum 0.03). With Guiv's package (his E13/E24/E43/E37/Q1) that is
   now every structure row either team has measured, at layers 20 to 60; it does state content (his SA-R, 204/396).
3. **NLA prompted: 8 of 10**, the two misses by small margins (HTML 0.84, quote 0.79).

## What failed, and what it says

4. **"NLA states iff used": 5 of 10.** Call argument (used) is stated 33%; CSV, heading, HTML (unused) are not
   stated, as predicted. But LaTeX math, the most strongly used row (z +22), is never stated; string state (z −12)
   is stated 3%; JSON depth (not used, z +2.6) is stated 25%; and language, which the rule predicted silent because
   it has no next-token consequence, is stated 95%. The NLA states what is *statement-sayable* (language, 0.51) or
   what it can phrase as a pending structure (a nested object "requires completion"), not what the model uses.
   "Used" is the wrong antecedent for statement; it was the right one on table column and parenthesis by
   coincidence of those two rows.
5. **"NLA never names the complement": 4 of 10.** It names the unmarked value freely when both values are positive
   states (first vs third argument 1.00, shallow vs deep JSON 0.85, level one vs three 0.96, first vs fourth field
   1.00, language 0.97) and almost never when the complement is an absence (outside a string 0.00, outside math
   0.00, outside a quote 0.04, one tag deep 0.21). The exp 17 one-sidedness is a property of open-span states, not
   of the NLA. Guiv's W1 shows the same split (parenthesis one-sided, table column both values named).
6. **"OLens exhibits": 6 of 10, all four misses on predicted-high rows.** The judge scored 0.00 on every structure
   row although the lens's bullets visibly reproduce nested JSON, open strings and math (`results/gen_ol_*.json`);
   the exhibits construct is reliable only for language (0.72 here), as exp 12 warned. Scored as misses; the cell
   should be dropped or re-judged with a structure-aware rubric before the rule is reused.
7. **J-lens: 2 of 10.** It reads eight rows at 0.74 to 1.00 where J-enrichment predicted low. The mined word sets
   show why: inside-math items contain `\sqrt`, argument items end on different tokens, CSV and JSON values differ
   in vocabulary. The synthetic banks carry lexical cues the J-lens can read, so "J-enriched ≤ 1.5 by class mean"
   is not the right J-lens predictor for them, and the uniform readable = 1.00 is partly the same cue. Bank design
   limitation, recorded.

## Reading for the thesis

The objective account survives on the reader whose objective we understand best: Oracle Lens reads what a
32-token span can re-induce and says nothing it cannot exhibit, on ten new rows and four layers. The NLA's
behaviour is not captured by "used": it states what is sayable by statement and names complements that are
positive states. The next version of the account should replace the NLA's antecedent with statement-sayability
plus a positive-state test, keep the Oracle Lens rules as they are, and build banks whose two values share
vocabulary before the J-lens rule is tested again. Nothing here changes exps 14 to 17; it bounds how far their
two rows generalise.

## Caveats

One model, one layer for the NLA; synthetic banks (lexically cued, see 7); judge single-pass (one cell parsed
300/320); prompted scores selected and evaluated on the same selection split (exp 07's convention, winner's curse
on the mined families); two rule clarifications after the first instrument run, both before any reader ran.
