# Exp 31: is "carrying" just copying the last 32 tokens? (copy overlap and copy-deleted carrying, laptop only)

**Question.** Claim A says both readers carry structural state in what they write (exp 25: Oracle Lens readout
0.84 to 0.97 and NLA explanation 0.93 to 1.00 through the critic on the string, math and depth rows of the public
bank; exps 20, 23, 27: a bag-of-tokens classifier on the readout text agrees). The boring explanation: the readers
reconstruct the local context from the activation and echo it, so a readout carries the state only because it
contains verbatim pieces of the last ~32 tokens, and it carries nothing beyond what a copy of that window carries.
Two things would follow: (i) readouts overlap their own item's window far more than a random other item's window
from the same row, and (ii) deleting everything that appears in the item's window from the readout kills carrying.
This experiment measures both on the stored k = 1 readouts. No GPU, no Modal, no LLM or judge calls.

**Data (all already on disk; nothing regenerated).**
- Public bank (primary): exp 22's selection split (`22_public_bank/results/items_base.json`, `split == "select"`,
  160 items per row, eight rows incl. the `code_lang` positive control) with exp 25's k = 1 readouts
  (`25_public_exhibits/results/ol_sel_{row}.json`, `nla_sel_{row}.json`). Label: `value == MARKED[row]` (inside,
  inside, inside, third, deep, deep, nested; `code_lang`: python).
- Exp 19 bank + exp 13 headline rows (replication): exp 20's readouts (`20_ar_structure/results/{ol,nla}_sel_{row}.json`)
  on `19_natural_bank` / `13_new_rows` selection items, ten rows, same MARKED convention as `figures/scripts/g4_s6_carry_ladder.py`.
- Random read points: exp 27's 1,939 points (`27_random_2000/results/items_base.json`, `ol27.json`, `nla27.json`),
  exp 23's binarised states (`analyze23.STATES`), per-kind cells with at least 20 items per class (the "all kinds"
  cells are skipped: they are confounded with document kind, as exp 27 says).
- Window = the item's last 32 Qwen3.6-27B token ids (`ids[-32:]`, the same tokens exp 20/25 fed the critic as
  `span_32`); the read point is the last id. Tokeniser: `Qwen/Qwen3.6-27B` from the local HF cache (offline).
- Per-item reconstructor outputs: `recon25_{row}.json` and `recon20_{row}.json` store only per-row AUROC, class-direction
  cosines and six example texts, no per-item probe projections. Measure (c) of `next.md` row F (items where the span
  rebuild fails but the readout rebuild succeeds) therefore cannot be computed without a GPU rerun and is skipped.

**Measures.** Readout text: as stored, with `<explanation>` / `</explanation>` tags removed for the overlap
measures only (the text-bag measures use the stored text exactly, so they reproduce exps 20/23/27).
- Normalised token: `tok.decode([t]).strip().lower()`; if that is empty (pure whitespace) the raw decoded string.
  A *content token* is one whose normalised form contains an alphanumeric character (Unicode `\w` minus `_`-only).
- (a) **Copy overlap.** Per readout: `tok_overlap` = share of its content tokens whose normalised form is among the
  normalised tokens of the window; `bi_overlap`, `tri_overlap` = share of its word 2-grams / 3-grams (words =
  lowercased `\w+` runs) that occur among the window's word n-grams (window text = `tok.decode(ids[-32:])`).
  Readouts with no content token / no n-gram are left out of that mean. **Shuffled baseline**: the same overlap
  against the window of a random other item of the same row (a fixed derangement, seed 31), so chance overlap
  includes the row's shared vocabulary. Reported per reader per row: mean own, mean shuffled, excess = own minus
  shuffled with a 95% bootstrap CI clustered by source file, and quartiles of own `tok_overlap`.
- (b) **Carrying after deleting the copied part.** The carrying measure is exactly exps 20/23/27's text bag:
  `lib19.bag_of_tokens` (binary, whole readout, min_df 3) on Qwen tokens of the readout, `mats.probes` logistic
  (lam 1.0) with grouped 5-fold CV by source file (seed 0), AUROC of the out-of-fold scores, 95% CI by cluster
  bootstrap over source files (2,000 resamples, paired across conditions so differences get CIs too). Conditions,
  each applied to the readout's token id list before the bag:
  - `full`: no deletion (reproduces exp 20's `readout_textbag.json` and exp 27's table; checked).
  - `del_own`: delete every readout token whose normalised form occurs in the item's own window (all tokens,
    punctuation included).
  - `del_bal` (leak-balanced, **primary**): delete tokens whose normalised form occurs in the own window OR in the
    window of one random item of the OPPOSITE class from the same row (seed 31). `del_own` alone can leak the label
    through which tokens are missing (e.g. a quote token deleted mostly from inside-string readouts); `del_bal`
    removes the same kind of material from both classes.
  - `del_rand` (deletion-size control): delete tokens occurring in the window of a random other item of the row
    (same derangement as the shuffled baseline): what deleting a comparable amount of non-copied text costs.
  - `del_row` (strict bound): delete every token whose normalised form occurs in ANY window of the row/cell.
  - `low_half`: the `full` bag restricted to the readouts at or below the row median of own `tok_overlap` (per
    reader), against the mean of 20 random halves of the same size (`rand_half`; seed 31..50), so the comparison is
    not confounded by halving the training data.
  - Reference: `window_bag` = the same classifier on the 32 window tokens themselves (what a perfect copy of the
    window would carry under this classifier).
- Main summary cells: the public bank's four best-carried rows `CARRY4` = py_string, js_string, latex_math,
  json_depth (exp 25's critic AUROC 0.84 to 1.00 for both readers); random points: `code (py+js) in_string`.

## Predictions written before any number was computed (hash in `results/predictions_sha256.txt`)

I have seen the six example readouts stored in `recon25_py_string.json` (Oracle Lens's bullets visibly reuse phrases
of the window, e.g. "Cannot cast", "must be provided w"); no overlap or AUROC in this experiment had been computed.

1. **Overlap is real and larger for Oracle Lens.** On the seven natural public-bank rows, mean own-window `tok_overlap`
   exceeds the shuffled baseline for both readers (excess CI above 0 on at least 6 of 7 rows for Oracle Lens), and
   Oracle Lens's mean excess over the seven rows is at least twice the NLA's (I expect roughly OL own 0.35 to 0.55 vs
   shuffled 0.15 to 0.30; NLA own 0.15 to 0.30 with excess under 0.10). Reading: if it holds, Oracle Lens (a
   continuation-style reader) does copy the window and the copying explanation is live for it; if Oracle Lens's
   excess is under 0.05, copying is not even a candidate and the rest of this experiment is moot.
2. **NLA carrying survives copy deletion.** On CARRY4, the NLA's `del_bal` text-bag AUROC is at least 0.70 on all
   four rows and its mean drop from `full` is at most 0.10. Reading: if it holds, the copying explanation is ruled out
   for the NLA on these rows (it carries the state in words not taken from the window: genre/format description);
   if `del_bal` falls to 0.60 or below on two or more rows, the NLA's text-level carrying on those rows is window
   copying and claim A must be narrowed accordingly for the text-bag evidence.
3. **Oracle Lens carrying is partly copying but not only.** On CARRY4, Oracle Lens's `del_bal` AUROC stays at least
   0.65 on at least 3 of 4 rows, and its mean drop from `full` is larger than the NLA's mean drop by at least 0.05.
   Reading: if it holds, Oracle Lens's carrying is part copy, part beyond-window (claim A survives, with the Oracle
   Lens numbers read as an upper bound on beyond-copy carrying); if `del_bal` falls under 0.60 on 3 or 4 rows, Oracle
   Lens's carrying of string/math/depth state is explained by copying and claim A holds for the NLA only.
4. **Low-overlap readouts carry as well, and random points agree.** (i) On CARRY4, for each reader, the mean `low_half`
   AUROC is within 0.08 of the mean `rand_half` AUROC; (ii) at random read points, `code (py+js) in_string` keeps a
   `del_bal` AUROC of at least 0.65 for both readers (full: 0.72 OL, 0.81 NLA in exp 27). Reading: if both hold,
   carrying does not depend on how much a readout copies; if `low_half` is more than 0.08 below `rand_half`, carrying
   is concentrated in the copying readouts (copy explanation supported for that reader).

Scoring rule: each prediction held / partial (some clauses hold) / failed, stated with the numbers. The critic
(frozen-probe) measure cannot be re-run on deleted readouts without a GPU; this experiment tests the copying
explanation against the text-bag evidence only, and says so in the reading.

---

## Results (30 Oct; laptop, CPU only, about 4 minutes; `analyze31.py` -> `results/table31.md`, `results/overlap31.json`, `results/carry31.json`; `score31.py` -> `results/score31.json`)

Everything above the separator is unchanged (sha256 `1fc10176...69af` in `results/predictions_sha256.txt`, re-checked
before this section was appended). Checks: the `full` text bag reproduces exp 20's `readout_textbag.json` (e.g.
py_string 0.959 / 0.952, py_arg 0.716 / 0.672) and exp 27's carrying table (e.g. code in_string 0.72 / 0.81) to three
decimals; the decoded tail of `ids` matched each item's text for every bank (read point = last id). Tests:
`test_lib31.py`, 12 passed. Measure (c) skipped: no per-item reconstructor outputs are stored (see Data).

### (a) Copy overlap (share of a readout's content tokens found in the item's last 32 tokens; shuffled = a random other item's window, same row)

Public bank (exp 25 readouts, 160 items per row; 95% CI of the excess clustered by source file):

| row | Oracle Lens own / shuffled | excess [CI] | NLA own / shuffled | excess [CI] | word 3-grams own (OL / NLA) |
|---|---|---|---|---|---|
| py_string | 0.21 / 0.04 | +0.17 [0.15, 0.19] | 0.14 / 0.04 | +0.10 [0.09, 0.11] | 0.01 / 0.01 |
| js_string | 0.22 / 0.03 | +0.19 [0.17, 0.21] | 0.12 / 0.03 | +0.10 [0.09, 0.11] | 0.01 / 0.01 |
| latex_math | 0.20 / 0.08 | +0.12 [0.10, 0.14] | 0.14 / 0.06 | +0.08 [0.07, 0.09] | 0.00 / 0.01 |
| py_arg | 0.19 / 0.04 | +0.15 [0.13, 0.17] | 0.09 / 0.02 | +0.07 [0.06, 0.08] | 0.00 / 0.00 |
| json_depth | 0.20 / 0.05 | +0.15 [0.12, 0.17] | 0.10 / 0.03 | +0.06 [0.05, 0.08] | 0.02 / 0.01 |
| md_heading | 0.20 / 0.05 | +0.15 [0.14, 0.17] | 0.14 / 0.05 | +0.09 [0.08, 0.10] | 0.01 / 0.01 |
| md_list | 0.24 / 0.04 | +0.20 [0.18, 0.22] | 0.17 / 0.04 | +0.12 [0.11, 0.13] | 0.01 / 0.01 |
| code_lang | 0.20 / 0.03 | +0.17 [0.15, 0.19] | 0.12 / 0.03 | +0.09 [0.08, 0.10] | 0.01 / 0.00 |

Word 2-gram overlap is 0.02 to 0.05 (OL) and 0.01 to 0.04 (NLA) against 0.00 shuffled. Oracle Lens own-overlap
quartiles are about 0.11 / 0.19 / 0.28 on every row. The exp 19 rows (exp 20 readouts) and exp 27's random points give
the same picture: Oracle Lens 0.17 to 0.27 own vs 0.02 to 0.11 shuffled, NLA 0.05 to 0.21 vs 0.01 to 0.10 (full
table in `results/table31.md`). So both readers do reuse window vocabulary well above chance, Oracle Lens about
twice as much as the NLA, but about four of five content tokens are not in the window, and verbatim phrases
(3-grams) are almost absent: the readers echo individual words, not stretches of text.

### (b) Text-bag carrying before and after deleting the window's tokens (AUROC [95% cluster CI])

Public bank, the four best-carried rows (CARRY4) plus the rest; `window bag` = the same classifier on the 32 window
tokens themselves; `del_bal` keeps about 0.65 to 0.72 of Oracle Lens's tokens and 0.79 to 0.83 of the NLA's;
`del_row` (every token seen in ANY window of the row) keeps only 0.24 to 0.29 (OL) and 0.30 to 0.45 (NLA):

| row | window bag | OL full | OL del_bal | OL del_row | NLA full | NLA del_bal | NLA del_row |
|---|---:|---|---|---|---|---|---|
| py_string | 0.72 | 0.90 [0.85, 0.94] | 0.87 [0.81, 0.92] | 0.72 [0.64, 0.79] | 0.92 [0.87, 0.96] | 0.92 [0.87, 0.96] | 0.89 [0.83, 0.93] |
| js_string | 0.56 | 0.84 [0.77, 0.90] | 0.81 [0.74, 0.87] | 0.68 [0.59, 0.76] | 0.90 [0.85, 0.95] | 0.89 [0.83, 0.94] | 0.77 [0.69, 0.84] |
| latex_math | 0.62 | 0.77 [0.71, 0.84] | 0.70 [0.62, 0.78] | 0.61 [0.53, 0.69] | 0.88 [0.83, 0.93] | 0.87 [0.81, 0.92] | 0.88 [0.82, 0.93] |
| json_depth | 0.99 | 0.97 [0.94, 0.99] | 0.96 [0.93, 0.98] | 0.94 [0.90, 0.97] | 0.98 [0.96, 0.99] | 0.98 [0.96, 0.99] | 0.97 [0.94, 0.99] |
| py_arg | 0.66 | 0.66 [0.58, 0.74] | 0.61 [0.53, 0.70] | 0.46 [0.37, 0.54] | 0.59 [0.50, 0.68] | 0.58 [0.49, 0.67] | 0.57 [0.48, 0.66] |
| md_heading | 0.84 | 0.71 [0.64, 0.79] | 0.67 [0.59, 0.75] | 0.68 [0.60, 0.76] | 0.61 [0.53, 0.69] | 0.59 [0.51, 0.68] | 0.58 [0.50, 0.66] |
| md_list | 0.84 | 0.56 [0.47, 0.64] | 0.49 [0.41, 0.59] | 0.49 [0.41, 0.58] | 0.59 [0.51, 0.68] | 0.62 [0.53, 0.70] | 0.64 [0.55, 0.72] |
| code_lang | 0.99 | 0.98 [0.96, 0.99] | 0.96 [0.92, 0.98] | 0.72 [0.64, 0.80] | 0.98 [0.96, 1.00] | 0.95 [0.92, 0.98] | 0.86 [0.80, 0.92] |

Paired drops full minus del_bal on CARRY4: Oracle Lens +0.03 [0.00, 0.05], +0.04 [0.00, 0.07], +0.08 [0.04, 0.12],
+0.01 [-0.01, 0.02]; NLA +0.00, +0.01, +0.02, +0.00 (all CIs include 0). The deletion-size control `del_rand` costs
0.00 to 0.03 on these rows. `del_own` (not leak-balanced) is within 0.03 of `del_bal` on CARRY4.

Low-overlap half vs random halves of the same size (mean over CARRY4): Oracle Lens 0.83 vs 0.85; NLA 0.91 vs 0.88.

Replication on exp 19's rows (exp 20 readouts), CARRY4: Oracle Lens full 0.96 / 0.88 / 0.78 / 0.94 -> del_bal 0.94 /
0.87 / 0.71 / 0.92; NLA 0.95 / 0.87 / 0.86 / 0.97 -> 0.94 / 0.84 / 0.85 / 0.96. Headline rows: table column NLA 0.91 ->
0.91, parenthesis NLA 0.81 -> 0.80 (Oracle Lens 0.71 -> 0.73 and 0.56 -> 0.52).

Random read points (exp 27), per-kind cells: del_bal is within 0.05 of full everywhere for the NLA (code in_string
0.81 -> 0.79 [0.74, 0.84]; tex in_math 0.85 -> 0.87; md list_depth 0.83 -> 0.83); Oracle Lens loses up to 0.08 (code
in_string 0.72 -> 0.66 [0.60, 0.71]; py in_string 0.60 -> 0.53; md list_depth 0.73 -> 0.67).

### Scoring the predictions

| prediction | result | verdict |
|---|---|---|
| 1. Overlap real and at least twice as large for Oracle Lens (OL excess CI > 0 on >= 6 of 7 rows; OL mean excess >= 2x NLA's) | CI above 0 on 7 of 7 rows for both readers; mean excess OL 0.161 vs NLA 0.089, ratio 1.81 (exp 19 rows: 1.83). Expected magnitudes were too high: OL own 0.19 to 0.24 (predicted 0.35 to 0.55), shuffled 0.02 to 0.08 (predicted 0.15 to 0.30) | partial (direction and significance hold, the 2x ratio fails at 1.8, levels lower than expected) |
| 2. NLA del_bal >= 0.70 on all CARRY4 rows, mean drop <= 0.10 | 0.92, 0.89, 0.87, 0.98; mean drop 0.010 (exp 19 rows 0.014) | held |
| 3. OL del_bal >= 0.65 on >= 3 of 4, and OL's mean drop exceeds NLA's by >= 0.05 | 0.87, 0.81, 0.70, 0.96 (4 of 4); mean drop 0.037 vs 0.010, difference 0.027 (exp 19 rows 0.017) | partial (survives, but copies less of its carrying than predicted) |
| 4. low half within 0.08 of random halves on CARRY4; code in_string del_bal >= 0.65 for both | OL -0.02, NLA +0.03; code in_string 0.66 (OL, CI [0.60, 0.71]) and 0.79 (NLA) | held (Oracle Lens at the line) |

### What it means for claim A

The copying explanation predicts that readouts overlap their own window far above chance and that deleting the
copied words kills carrying. The first half is true (both readers reuse window words at 3 to 6 times the chance
rate, Oracle Lens about twice the NLA's), but the second is not: deleting every token type that occurs in the item's
window, even with the leak-balanced deletion, leaves the NLA's text-level carrying unchanged on every
string/math/depth row (drops of 0.00 to 0.02) and costs Oracle Lens 0.01 to 0.08, the largest on LaTeX math (0.77 ->
0.70). Two further facts point the same way: the readouts' text bag beats the window's own text bag on the string and
math rows (e.g. JavaScript string 0.84 to 0.90 against 0.56), so a bag of copied window tokens could not have
produced these numbers, and verbatim 3-gram overlap is about 0.01, so ordered copying of window phrases is not the
channel either. So for the text-bag evidence, carrying is not window copying, for either reader, on the rows that
carry; a small part of Oracle Lens's carrying (up to about a third of its margin over 0.5 on LaTeX math and
Python-string random points) does sit in copied words. This does not test the critic measure itself (the
0.84 to 1.00 frozen-probe AUROCs cannot be recomputed on deleted readouts without a GPU), and it says nothing about
the other open alternatives for claim A (state-correlated genre or content vocabulary in natural banks, the critic's
in-distribution advantage for the NLA).

### Limitations

- Deletion is by normalised token type (stripped, lowercased decoded Qwen token). A readout that re-tokenises a
  window word differently (e.g. `ValueError` vs `Value` + `Error`), paraphrases it, or copies from beyond the last 32
  tokens is not caught; `del_row` (deletes every token type seen in any window of the row, 55 to 76% of the readout)
  is the conservative bound and still leaves the NLA at 0.77 to 0.97 and Oracle Lens at 0.61 to 0.94 on CARRY4.
- The n-gram overlap uses `\w+` words, so punctuation-anchored fragments (an open quote before a word) are not counted
  as n-grams; they are counted in the token measure.
- The text bag is order-blind; that is why the 3-gram measure is reported alongside it. Exp 25's critic measure
  (ordered) is not re-run.
- The low-half split is not class-balanced (e.g. 34 of 80 positives on py_string for both readers): own-window
  overlap is mildly higher on the marked class. Several random-point cells are small (tex in_math 22 positives,
  in_string 51 per kind), and the critic's per-item scores are not stored, so measure (c) is not available.
