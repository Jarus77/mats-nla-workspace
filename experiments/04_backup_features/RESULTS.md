# 04_backup_features: backup non-J candidates on Qwen3.6-27B

Status 2026-10-06. Same gate as 02 (D9 final G2, D11 centred steering), so the two directories are
comparable.

- **Layers.** L = output of decoder block L = HF `hidden_states[L+1]`. L42 is primary, L40/L44 are
  sensitivity, and L38 is the steering layer.
- **J-space.** The top-r right singular vectors of W_U·diag(1+w)·J_L, from neuronpedia n1000. Bases come
  from `/results/03_pipeline_smoke/jspace_basis_L*.pt` (overlap 1.0000 with 02's).
- **Ranks.** r\* = 384 and 4r\* = 1536.
- **Cross-check lens.** camilablank (25 prompts), with its own basis (overlap with n1000 at r\* is 0.81).

Machine-readable output: `results/gates.json`. Raw JSONs are in `/results/04_backup_features/{gate,causal}`.

## TL;DR: no backup qualifies as a clean NON-J feature

**None of the 9 features passes every gate.** `col` remains the only clean non-J feature.

**P1, non-count candidates (to defeat the "col is special because it's a count" objection).** All four fail:

| feature | outcome | evidence |
|---|---|---|
| capitalisation | IN-J | Next word capitalised, mid-sentence: enrichment 2.71 at r\*. The J-Lens top-10 is visibly enriched for capitalised tokens (AUROC 0.81). It is causal, 7× random at L38. |
| coding language (Py vs JS) | DIFFUSE-BUT-J-READABLE | Enrichment 1.49/1.33 is borderline ≤1.5. The J-Lens label words track it (AUROC 0.71 vs a band upper edge of 0.51). G4 passes at 5.5× (p90) / 10× (p99), but the J-removed steer **leaks** into the J-Lens at 40/42/44 (z 2.4–5.8). |
| in-quote state | IN-J | Enrichment 2.22/1.70. Lens readable. G4 at L38 is only 2.6×. |
| French gender agreement (upcoming noun, read at gender-neutral *les/des/ces…*) | fails G1, and J-enriched | Probe AUROC 0.60 fails G1. Enrichment 4.15. Lens weakly readable. |

**P2, count replications.**

- **Indentation level** is the only feature that is geometrically and functionally non-J:
  - enrichment 1.11/1.14 (below the shuffled-label 1.25/1.17);
  - J-Lens not readable at L42/44;
  - G1 0.965.
- **But it fails G4.** Steering the J-removed direction (and even the raw direction) does not move
  next-line indentation in the predicted direction (×0.6 random, z −1.4). A post-hoc metric (8- vs
  4-space indent) does not move either (×0.1).
- Label: **"decodable, unused"** (prereg appendix).
- **Words since sentence start** is J-depleted geometrically (enrichment 1.03/1.09) but **J-Lens-readable**:
  - label words 0.513 and top-10 sentence-end punctuation 0.531, both above band;
  - G4 at L38 is 3.7× at p99 (below 5×);
  - the J-removed steer leaks at 40/42/44 at p99.
- Label: DIFFUSE-BUT-J-READABLE.

**P3, the in-J count calibration (list index with numerals in context) FAILED.** It is not J-Lens-readable:

- Digits never enter the lens top-10, even at the token right before "Item N" is emitted.
- Its enrichment is no higher than its own shuffled-label baseline (2.45 vs 2.53).
- Steering the index direction has no effect (×0.4 random).
- This matches 02's `col_report`: **numbers do not enter Qwen's J-space at L42 here.**
- So there is still no in-J count calibration.

**Comparability warning for `col` (new).** I applied Fable's emitted-form functional test to 02's `col`.

| emitted form | statistic | value | null band | verdict |
|---|---|---|---|---|
| newline tokens in the lens top-10 | AUROC | 0.501 | [0.498, 0.502] | **not readable** |
| soft newline mass (full distribution) | Spearman with col | 0.13 | [−0.015, 0.015] | **readable** |
| | AUROC | 0.58 | | |

- I therefore use **label words + top-10 category counts** (Fable's "classify the top-10 tokens") as the
  primary functional rule.
- The soft-mass variant is reported as a stricter supplement. **Even `col` fails it**, because the J-Lens
  partly behaves like a next-token lens: any causally used feature tilts its full distribution.

## Gate table (L42; G4 at L38, centred steering)

| feature | role | items / rows | G1 AUROC [95% CI] | enrichment r\*/4r\* (shuffled-label ref) | J-Lens label-word AUROC [null 95%] | emitted top-10 / soft AUROC [soft null] | label n1000 | label camilablank | G4 @L38, J-removed(1536): effect, ×max \|random\|, z vs J-complement (best dose) | leak 40/42/44 at that dose | self-report AUROC | **qualifies** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap_next | P1 | 1173 / 9280 | 0.864 [0.856, 0.871] | **2.71** / 1.49 (2.17 / 1.46) | 0.504 [0.494, 0.505] | **0.806** / 0.830 [0.49, 0.51] | IN-J | IN-J | p90 +0.187, 7.3×, z 16 ✓ | 42, 44 (negative) | 0.69 | no |
| cap_next_all | P1 variant | 1200 / 9593 | 0.901 [0.894, 0.906] | **1.75** / 1.26 (2.26 / 1.48) | 0.492 (below) | **0.791** / 0.793 | IN-J | DIFFUSE-READABLE | p90 +0.523, 12.5×, z 30 ✓ | none | 0.66 | no |
| code_lang | P1 | 326 / 2575 | 0.999 [0.999, 1.000] | 1.49 / 1.33 (1.26 / 1.21) | **0.707** [0.487, 0.513] | 0.488 / 0.547 [0.48, 0.52] | DIFFUSE-BUT-J-READABLE | DIFFUSE-READABLE | p90 +0.277, 5.5×, z 12 ✓ | **40, 42, 44** (z 2.4–5.8) | 1.00 | no |
| in_quote | P1 | 2368 / 18872 | 0.998 [0.998, 0.999] | **2.22 / 1.70** (2.05 / 1.52) | **0.509** [0.496, 0.504] | **0.526** / 0.744 | IN-J | IN-J | p99 +0.334, 2.6×, z 7 ✗ | none | 1.00 | no |
| gender_next | P1 | 1694 / 7407 | **0.600** [0.588, 0.612] ✗ | **4.15 / 1.84** (2.54 / 1.47) | **0.513** [0.495, 0.504] | **0.539** / 0.563 | IN-J | IN-J | p99 +0.038, 2.8×, z 7 ✗ | none | 0.50 | no |
| indent | P2 | 382 / 2967 | 0.965 [0.956, 0.974] | **1.11 / 1.14** ✓ (1.25 / 1.17) | 0.474 (below band [0.490, 0.509]) | 0.512 [0.485, 0.515] / 0.458 (below) | **NON-J** (strict: anti-tracks) | NON-J | p90 −0.007, 0.6×, z −1.4 ✗ (raw dir also negative, z −9) | none | 0.83 | no: **decodable, unused** |
| sent_pos | P2 | 1186 / 7428 | 0.971 [0.967, 0.974] | **1.03 / 1.09** ✓ (2.24 / 1.53) | **0.513** [0.495, 0.504] | **0.531** / 0.575 | DIFFUSE-BUT-J-READABLE | DIFFUSE-READABLE | p99 +0.259, 3.7×, z 9 ✗ (L42: 7.7×) | 40, 42, 44 (z 1.8–4.4) | 0.96 | no |
| list_num (mid-item) | P3 in-J calib. | 405 / 662 | **0.773** [0.728, 0.811] ✗ | 2.37 / 1.55 (2.02 / 1.48) | 0.483 [0.485, 0.514] | 0.500 (no digits in top-10) / 0.464 | J-ENRICHED-NOT-READABLE (≈ shuffled) | same | +0.001, 0.0× ✗ | none | 1.00 | calibration FAILED |
| list_hdr ("Item N" header) | P3 in-J calib. | 247 / 272 | 0.875 [0.835, 0.913] | 2.45 / 1.36 (2.53 / 1.44) | 0.491 [0.475, 0.522] | 0.500 / 0.442 | J-ENRICHED-NOT-READABLE (≈ shuffled) | same | +0.014, 0.4× ✗ | none | 1.00 | calibration FAILED |

How to read the table:

- **Bold** marks the failing criterion.
- **Rules:**
  - G1: AUROC ≥ 0.85.
  - G2-geometric: diff-in-means enrichment ≤ 1.5 at both r\* and 4r\*.
  - G2-functional (primary): one-sided, as in 02 s3d. The feature counts as J-readable if the label-word
    statistic or the emitted top-10 statistic is above its 200-permutation 95% band.
  - G4: the J-removed (1536) direction at L38 under centred steering h − (v·h)v ± Δv. Pass requires
    ≥5× the max |effect| of K=24 J-complement random directions, z ≥ 3, and item-paired Wilcoxon
    p < 0.01 vs the strongest random direction.
  - Leak: J-Lens AUROC or mean shift outside the two-sided J-complement band.
- **"strict"** adds below-band (anti-tracking) and soft-mass readability. Under it, indent becomes
  DIFFUSE-BUT-J-READABLE, because the lens *anti*-tracks it (label 0.474, soft 0.458, both below band).

## Details

### G1 by layer (AUROC, grouped 5-fold, cluster-bootstrap CI over items/problems)

| feature | L38 | L40 | L42 | L44 |
|---|---|---|---|---|
| cap_next | 0.855 | 0.861 | 0.864 | 0.874 |
| code_lang | 0.999 | 0.999 | 0.999 | 0.998 |
| in_quote | 0.998 | 0.998 | 0.998 | 0.999 |
| gender_next | 0.591 | 0.600 | 0.600 | 0.597 |
| indent | 0.967 | 0.961 | 0.965 | 0.965 |
| sent_pos | 0.965 | 0.965 | 0.971 | 0.980 |
| list_num | 0.730 | 0.745 | 0.773 | 0.753 |
| list_hdr | 0.852 | 0.847 | 0.875 | 0.885 |

Text baselines (held-out AUROC):

| feature | bag of the last 4 tokens | absolute position alone |
|---|---|---|
| code_lang | 0.94 | — |
| in_quote | 0.96 | 0.59 |
| indent | 0.90 | **0.71** (level-2 lines sit later in snippets; not position-matched) |
| sent_pos | 0.82 | 0.50 (position-matched by design) |
| cap_next | 0.64 | 0.48 |
| gender_next | 0.56 | 0.51 |

### G2-geometric: enrichment by layer (diff-in-means; random = 1.0)

- **indent** is the only feature that stays ≤1.5 at every read layer:
  - L40: 1.22 / 1.23
  - L42: 1.11 / 1.14
  - L44: 1.30 / 1.08
  - L38: 1.73 at r\*(38) = 192, then 1.35 / 1.33 at 384 / 1536.
- **sent_pos** is depleted at L42/44 (1.03, 0.95–1.09) but not at L40 (1.45 / 1.35) or L38 (2.34 at 192).
- The other features are enriched at all layers:
  - cap_next 2.7–3.2 at r\*;
  - in_quote 2.2–2.8;
  - gender 4.2–5.3;
  - code_lang 1.49–1.69 (borderline).
- **Shuffled-label reference (new, recommended).** Take the diff-in-means of a *random* relabelling. It
  is already enriched 1.2–2.5× at r\*, because it points along high-variance directions, and generic
  activations are 2.2× enriched (02).
  - For small or weak features (lists, gender) the real enrichment is no larger than this reference, so
    their enrichment is uninformative.
  - indent (1.11 vs 1.25) and sent_pos (1.03 vs 2.24) are clearly *below* their references.
- **INLP top-k enrichment** is 0.6–1.6 for every feature, as in 02 (non-discriminative).

### G2-functional

- **Label words**: 02 s3d replica; −log10 rank in softmax(W_U·norm(J·h)); mean one-vs-rest AUROC.
- **Emitted form** (Fable round 4). The tokens the lens would emit for each class:
  - capitalised vs lowercase word tokens;
  - Python vs JS keywords;
  - closing vs opening quote tokens;
  - feminine vs masculine noun/adjective lexicon (corpus-built, ≥90% purity);
  - 8+ vs ≤4-space runs;
  - sentence-final punctuation;
  - index digits.
- The top-10 category count is primary, and the soft log-mass is a supplement.
- Lens-list decodability (bag of the top-25 lens tokens, 02 s7 style; AUROC):
  - cap_next 0.78 (vs local-4 0.64), so the lens adds information;
  - in_quote 0.83;
  - code_lang 0.78;
  - indent 0.74 (vs local-4 0.90);
  - sent_pos 0.59;
  - lists 0.55–0.57.
- Most common L42 lens tokens at indentation positions include ` number`, ` count` and ` numbers` (a
  generic "counting" concept, not the value).
- **camilablank cross-check** agrees with n1000 on every label except two:
  - cap_next_all becomes DIFFUSE (its enrichment is 1.12 under that basis);
  - indent stays NON-J, but its soft mass is above band (0.53).

### G2-secondary: J-only probe vs 10 random equal-rank subspaces (L42; held-out d′, log-loss)

| feature | d′ J-only vs random mean (z), r\*=384 | 4r\*=1536 |
|---|---|---|
| cap_next | 1.78 vs 1.70 (+4.8) | 1.68 vs 1.63 (+3.5) |
| in_quote | 3.61 vs 3.51 (+2.0) | 4.40 vs 4.44 (−0.9) |
| gender_next | 0.51 vs 0.44 (+2.9) | 0.44 vs 0.42 (+1.0) |
| code_lang | 2.66 vs 2.70 (−0.3) | 3.92 vs 3.92 (−0.1) |
| indent | 1.47 vs 1.67 (**−3.6**) | 2.21 vs 2.29 (−1.4) |
| sent_pos | 1.66 vs 2.11 (**−18.7**) | 2.31 vs 2.47 (−3.6) |
| list_hdr | 0.79 vs 1.26 (−3.6) | 1.26 vs 1.70 (−6.0) |

- The count features (indent, sent_pos, list) are worse than random from inside J, as `col`/`depth` were
  in 02.
- The P1 features are at or above random.

### G4: causal use (centred steering, all context positions, 260 items, 2-fold cross-fit by group)

Signed effect = (m(+Δ) − m(−Δ))/2 in metric units. Doses are p90/p99 of |proj − median|; ‖Δ‖/‖h‖ is
0.09–0.21 at L38. Nulls are 24 J-complement directions plus 20 isotropic directions, all at the
jrem1536 p99 norm.

**L38** (each cell: effect, then ×max|J-complement random|):

| feature | metric | jrem1536 p90 | jrem1536 p99 | jrem(r\*=192) p99 | raw p99 | KL jrem1536 p99 / random |
|---|---|---|---|---|---|---|
| cap_next | log P(" Upper") − log P(" lower") | +0.19, 7.3× | +0.28, 10.8× | +0.79, 31× | +1.32, 51× | 0.069 / 0.009 |
| code_lang | log P(py keywords) − log P(js keywords) | +0.28, 5.5× | +0.50, 10× | +2.13, 43× | +2.19, 44× | 0.004 / 0.002 |
| in_quote | log P(close-quote) − log P(open-quote) | +0.17, 1.3× | +0.33, 2.6× | +1.19, 9.2× | +1.65, 13× | 0.072 / 0.011 |
| gender_next | log P(fem lexicon) − log P(masc lexicon) | +0.024, 1.8× | +0.038, 2.8× | +0.16, 12× | +0.33, 25× | 0.051 / 0.008 |
| sent_pos | logit P(. ! ?) | +0.16, 2.3× | +0.26, 3.7× | +1.07, 15× | +1.41, 20× | 0.044 / 0.008 |
| indent | E[next-line indent] (pre-registered) | −0.007, 0.6× | −0.016, 1.4× (z −3.6) | −0.046, 4× (z −10) | −0.041, 3.6× (z −9) | 0.012 / 0.003 |
| indent | post-hoc: log P(8-space) − log P(4-space) | +0.011, 0.1× | +0.005, 0.1× | −0.093, 1.1× | −0.098, 1.2× | — |
| list_num | E[next index digit] | +0.001, 0.0× | — | — | — | — |
| list_hdr | E[next index digit] | +0.014, 0.4× | — | — | — | — |

- **Retained fraction (jrem1536 / raw, p99)** is the 02 diagnostic:
  - code_lang 23%, cap_next 21%, in_quote 20%, sent_pos 18%, gender 12%;
  - 02's `col` was 79%.
- So for every P1 feature the causal route runs mostly through the top-1536 J-space.
- At L42, jrem1536 p99 reaches 5–18× for cap_next / code_lang / in_quote / sent_pos. That is reported
  but not gated.

**Leak (steer L38, J-Lens label contrast at 40/42/44, two-sided J-complement band, D11):**

- code_lang leaks at all three layers (AUROC 0.58–0.66 vs band upper edges 0.54–0.56; z 2.4–5.8).
- sent_pos leaks at 42/44 at p90 and at 40/42/44 at p99 (z 1.8–4.4).
- cap_next anti-leaks (0.44–0.46, below band).
- indent, in_quote, gender and the lists show no leak.

### Self-report (logit 2AFC at the probed position, chat; reported, not gated)

| feature | AUROC |
|---|---|
| code_lang | 1.00 |
| in_quote | 1.00 |
| list_num / list_hdr | 1.00 (text-computable) |
| sent_pos | 0.96 |
| indent | 0.83 (answers "No" on every item; the AUROC comes from the logit margin) |
| cap_next | 0.69 |
| gender_next | 0.50 |

## What this means for the go/no-go

1. **Backups do not add a second clean non-J feature.**
   - Indentation is the closest: J-depleted at every read layer, not J-Lens-readable at L42/44, worse than
     random from inside J. But the model does not causally use its J-removed direction for indentation
     behaviour. It looks like "decodable, unused", like 02's weak `depth` result.
   - If a second feature is mandatory, indentation is the candidate for a better-designed G4. Indent tokens
     are emitted after the newline, so the behaviour may be governed by copying the previous line's
     whitespace, not by this direction.
2. **The pattern from 02 holds and sharpens.** The count/positional features (indent, sent_pos, col,
   depth) are geometrically J-depleted. Every non-count feature tested (capitalisation, coding
   language, quote state, gender agreement) is J-enriched or J-Lens-readable, and its causal route
   runs ≥77% through J.
   - That is the opposite of what the "col is special because it's a count" objection needed. On Qwen L42,
     natural non-J content appears to be **positional bookkeeping**.
   - This is a reportable finding, but it means we cannot answer the objection with a non-count feature.
3. **The in-J count calibration fails, for col_report and both list variants.** Numerals in context
   never reach the J-Lens. Use language/tense (02) as the in-J calibration instead.
4. **Methodological notes for the prereg:**
   - (a) Report a shuffled-label enrichment reference next to every enrichment.
   - (b) The soft emitted-mass functional test flags even `col`, so do not use it as the gate.
   - (c) G4 with centred steering plus the D11 leak test kills code_lang, which would otherwise pass G4.

## Caveats

- **Single dataset per feature; linear probes only.**
  - indent is not position-matched (absolute position alone gives 0.71).
  - list_hdr has only 272 rows after position matching.
  - gender ground truth = spaCy morphology of the actual head noun, not the model's prediction (G1 0.60).
- **G4 metrics are hand-built next-token contrasts.** The indent metric failed both a pre-registered and
  a post-hoc version (both reported). G4 uses 260 items per feature.
- **G1 protocol differs from 02.** It uses grouped 5-fold CV with AUROC (02 used 3 grouped 80/20 splits).
  Everything else (J-space basis, enrichment, the label-word functional test, the 200-permutation band)
  is 02's procedure.
- **One-sided vs strict functional rule.** The one-sided rule follows 02 (`depth` failed at 0.5035). Under
  the strict two-sided rule, indent's below-band anti-tracking would also exclude it.
- **Compute.** About 20 H100-hours on Modal, roughly $80 at list price.

## Files

- **Code** (`experiments/04_backup_features/`):
  - `env.py`: Modal images (03 pins, plus datasets and spaCy fr_core_news_md).
  - `bf_feats.py`: feature specs and label words.
  - `bf_items.py`: item builders, plus position matching.
  - `bf_vocab.py`: emitted-form token sets.
  - `bf_gate_lib.py`: probes, enrichment, INLP, the J-Lens readout, the functional tests, J-only probes.
  - `bf_modal.py`: stages data / capture / gate. Run `modal run bf_modal.py --stage data|capture|gate [--only ...]`.
  - `bf_causal.py`: G4 with centred steering, the leak check and self-report. Run `modal run bf_causal.py [--only f] [--layers 38,42] [--behaviour indent8_vs_4]`.
  - `bf_colcheck.py`: the emitted-form test on 02's `col` (read-only use of 02's captures).
  - `summarize.py`: local; writes `results/gates.json` and the table.
- **Volume** (`/results/04_backup_features/`):
  - `data/*.pkl`
  - `acts/<dataset>/L{38,40,42,44}.npy`
  - `gate/<f>.json`, `gate/<f>_dirs.pt`
  - `causal/<f>_L{38,42}[ _indent8_vs_4].json` and `_raw.pt`
  - `colcheck/col_emitted.json`
