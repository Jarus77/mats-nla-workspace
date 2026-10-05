# 02_nonj_features: natural features outside the J-space on Qwen3.6-27B (ground truth for source attribution)

Status 2026-10-05. Everything is at **L42 = output of decoder block 42 = HF `hidden_states[43]`** (lead's K0
decision). L40 and L44 (block outputs 40/44 = `hidden_states[41]`/`[45]`) are sensitivity checks, and
`hidden_states[42]` is a sanity row. Machine-readable gate table: `gates.json` (also at
`/results/02_nonj_features/gates.json`).

## TL;DR

- **One clean non-J feature qualifies: character count since the last newline (`col`).** It is decodable
  (R² 0.655), J-depleted (diff-in-means enrichment 0.67 at r\*, 1.10 at 4r\*), its value tokens are not
  readable from the J-lens (paired AUROC 0.500, null band [0.499, 0.501]), and it is causally used:
  steering along the J-removed (4r\*) direction shifts newline log-odds by +0.99 at q90, against at most
  0.02 for random directions (49x). Steering at L38 shows no leak into the J-lens.
- **`col_report` also passes every gate, but it is the same feature in a chat frame.** It was meant to be
  the in-J calibration IJ1 (count under an explicit "how many characters" question). On Qwen it stays out
  of the J-space, so **the IJ1 calibration failed**: Qwen does not pull the count into J on demand (see the
  replication below).
- **Bracket depth (`depth`) is borderline.** It is geometrically non-J (0.57/1.16) and causal, but weakly
  (J-removed steering 1.5-1.9x random). It fails the functional criterion by a negligible margin (AUROC
  0.503 vs a band upper edge of 0.501, n=40k), and the J-lens top-25 list predicts depth through
  structural tokens (lens-bag R² 0.62 vs 0.35 from the last 4 tokens). Label: DIFFUSE-BUT-J-READABLE, so it
  is excluded from the clean set.
- **Predicted POS (`pred_pos`) and tense fail the new geometric gate.** Their diff-in-means directions are
  J-enriched (POS 2.62 at r\*; tense 5.6 at r\* and 2.3 at 4r\*). Tense is also weakly J-readable (AUROC 0.52
  > band). Both would have passed the old test (probe survives J-removal, label words absent from the lens).
  **The standard non-J test is uninformative**: every feature survives J-removal, including the in-J
  calibrations, by the same margin as an equal-rank random removal.
- **In-J calibrations behave as expected on the new gate.** Topic: enrichment 5.98/2.34, J-lens AUROC 0.83.
  Language: 3.32/1.73, AUROC 0.62. Both "survive J-removal" exactly as well as the non-J features do.

## Gate table (L42)

r\* = 384 (smallest rank whose rebuilt J-lens keeps ≥90% top-10 agreement on WorkspaceBench basic_readout
at J-layer 42; 4r\* = 1536). J-space at rank r = top-r right singular vectors of M = W_U·diag(1+w_norm)·J_42
(neuronpedia n1000 lens).

| feature | role | G1 probe @L42 (ctrl task) | G2-geom: diff-means enrichment r\*/4r\* (≤1.5) | G2-func: J-lens AUROC [null 95%] | G4: J-removed(4r\*) steer q90 / q99 vs max random | leak @L38→40/42/44 | old G2: retention J − rand (r\*/4r\*) | **label** |
|---|---|---|---|---|---|---|---|---|
| col | NJ1 | R² 0.655 [0.626, 0.674] (0.12) | **0.67 / 1.10 ✓** | **0.500 [0.499, 0.501] ✓** | **+0.99 / +1.47 vs 0.02 / 0.03 ✓** | none ✓ | +0.005 / +0.016 | **NON-J** |
| col_report | IJ1 (expected in-J) | R² 0.880 [0.867, 0.895] (0.13) | 0.59 / 0.86 ✓ | 0.500 [0.498, 0.502] ✓ | +0.74 / +1.01 vs 0.21 / 0.35 ✓ | none | +0.001 / +0.003 | NON-J (calibration FAILED) |
| depth | NJ2 | R² 0.945 [0.939, 0.949] (0.949*) | 0.57 / 1.16 ✓ | 0.5035 [0.499, 0.501] ✗ (negligible) | +0.047 / +0.070 vs 0.025 / 0.047 (weak) | noisy, inside the random range | 0.000 / 0.000 | DIFFUSE-BUT-J-READABLE (borderline) |
| pred_pos | NJ3 | acc 0.823 [0.811, 0.832] (0.54), chance 0.125 | **2.62** / 1.56 ✗ | 0.481 (below band) | +0.10 / +0.18 vs 0.009 / 0.016 ✓ | none | +0.005 / −0.013 | J-ENRICHED, value tokens not J-readable |
| tense | NJ4 | acc 0.845 [0.816, 0.866] (0.70), chance 0.5 | **5.60 / 2.31** ✗ | 0.520 [0.495, 0.506] readable | +0.31 / +0.59 vs 0.047 / 0.069 ✓ (but 15-19% of full effect) | none | +0.005 / −0.013 | IN-J |
| lang | IJ2 | acc 0.966 [0.958, 0.973] (0.32) | 3.32 / 1.73 | 0.625 readable | +0.003 / +0.008 (≈0; works only via J) | none | +0.003 / 0.000 | IN-J |
| topic | in-J reference | acc 0.849 (0.55) | 5.98 / 2.34 | 0.830 readable | not run with J-removed direction | – | −0.009 / −0.019 | IN-J |
| nexttok | in-J reference | acc 0.720 (0.47) | 1.29 / 1.54 | 0.566 readable | not run with J-removed direction | – | +0.004 / −0.017 | intermediate, J-readable |

\* The Hewitt & Liang control task is degenerate for `depth`. The brackets vocabulary is tiny, so a
probe memorises one random label per token type (control R² 0.949). A better selectivity check is the
token-identity baselines from `s7_lensbag`: current token alone R² 0.20, bag of the last 4 tokens R² 0.35,
both far below 0.945. The same problem gives `next_pos` a control (0.58) above its real accuracy (0.53). We
replaced that feature with `pred_pos` per the lead.

Doses and ‖Δ‖/‖h‖ (median residual norm, L42): q90 / q99 =
col 0.11 / 0.16, depth 0.23 / 0.39, pred_pos 0.11 / 0.19, tense 0.11 / 0.16, lang 0.14 / 0.23,
col_report 0.14 / 0.24, topic 0.15 / 0.22. 2×q99 is also recorded but is in the NLA-FVE void. Depth's
doses are large relative to ‖h‖.

## Details per criterion

### G1: linear probe (3 grouped-by-document 80/20 splits, cluster-bootstrap CI)

| feature | L40 (hs41) | L42 (hs43) | L44 (hs45) | hs42 sanity |
|---|---|---|---|---|
| col R² | 0.585 | 0.655 | 0.653 | 0.592 |
| depth R² | 0.941 | 0.945 | 0.949 | 0.943 |
| pred_pos acc | 0.821 | 0.823 | 0.835 | 0.818 |
| tense acc | 0.842 | 0.845 | 0.853 | 0.844 |
| lang acc | 0.959 | 0.966 | 0.972 | 0.959 |
| col_report R² | 0.816 | 0.880 | 0.891 | 0.831 |

- `col` ordinal accuracy (10-character bins) is 0.35, within one bin 0.80, Spearman 0.85. A linear probe
  only partly captures the count (Gurnee et al. report a curved manifold).
- Block 42 adds about 0.06 R² over block 41 for both count features.
- Class sizes: 4000 per class (topic, POS, tense, language); 1430-4000 for nexttok; regression capped at
  40k rows.

### G2-geometric: primary (lead, final)

- We take the enrichment ‖Q_rᵀd‖²/(r/d) of the diff-in-means (classes) or covariance (counts) direction d.
  Random = 1.0.
- Centred activations themselves are enriched 2.2x at r\* and 1.5x at 4r\*, so "≤1.5" means "less
  J-loaded than a generic activation direction".
- INLP top-8 enrichments (`gates.json`, `jonly/*.json`) are 0.5-1.1 for every feature, in-J ones included.
  Logistic/INLP directions mostly pick low-variance noise directions and do not discriminate, which agrees
  with the lead's choice of diff-in-means.
- Sensitivity (diff-in-means at r\*): col 0.98/0.49 (L40/L44), depth 0.79/0.56, pred_pos 3.12/2.34,
  tense 5.62/5.30, lang 3.35/2.92, topic 5.78/5.60.

### G2-functional: does the full J-lens readout track the feature?

- Test: J-lens (standard readout softmax(W_U·norm(J·h)), n1000) rank scores of the value-naming tokens.
  We used number words and digits for counts; label words (adjective/名词/past/Russian/俄语/...) for
  classes.
- Statistic and null: AUROC against a permutation null (`s3d_functional.py`).
- Results (top-k presence, `probe_results/*.json → lens`):
  - col and col_report: a number token is in the top-10 at 1.4% / 0.15% of positions; the correct value
    appears no more often than a permuted value.
  - POS labels: in the top-50 at 0.02%.
  - tense labels: in the top-10 at 0.2%.
  - topic labels: in the top-10 at 13%.
  - language names: in the top-10 at 1.3%.
  - L40/L44 give the same picture (`probe_results_sens/`).
- Lens-list decodability (`s7_lensbag`; bag of the top-25 lens tokens vs bag of the last 4 tokens):
  - topic: 0.69 vs 0.52, so the lens adds information.
  - depth: R² 0.62 vs 0.35, so the lens adds information, through bracket and quote tokens.
  - pred_pos: 0.45 vs 0.36. The lens list contains the predicted next words.
  - tense: 0.665 vs 0.669.
  - lang: 0.34 vs 0.96.
  - col_report: 0.15 vs 0.35.
  - col (RidgeCV): lens bag R² −0.03 vs 0.38 from the last 4 tokens; adding the lens to local4 gives
    0.40. The J-lens list carries no character-count information.

### G2-secondary: J-only probe vs 10 random equal-rank subspaces (accuracy / R², z vs random)

| feature | r\*=384: J-only vs random mean (z) | 4r\*=1536 |
|---|---|---|
| col | 0.268 vs 0.389 (−14.6) | 0.524 vs 0.551 (−4.0) |
| depth | 0.832 vs 0.875 (−17) | 0.921 vs 0.925 (−6.7) |
| col_report | 0.449 vs 0.683 (−36) | 0.784 vs 0.827 (−11) |
| pred_pos | 0.705 vs 0.721 (−3.6) | 0.822 vs 0.811 (+5.2) |
| tense | 0.815 vs 0.780 (+4.8) | 0.848 vs 0.831 (+4.5) |
| topic | 0.841 vs 0.789 (+9.3) | 0.851 vs 0.842 (+2.7) |
| lang | 0.765 vs 0.839 (−20) | 0.944 vs 0.944 (0.2) |
| nexttok | 0.624 vs 0.661 (−9.9) | 0.742 vs 0.728 (+11) |

- The count features are worse than random from inside the J-space, which is consistent with non-J.
- Topic and tense are better than random.
- Language is not, although its mean-diff direction is enriched. This test is noisy, as the lead
  expected.
- Caveat: we measured accuracy / R², not log-loss or d′. Only lang (0.94-0.97) is near saturation.

### Old G2 (probe survives J-removal), shown for completeness: uninformative

- Retrained probes keep ≥98% of baseline at r\* and 4r\* for **every** feature, in-J calibrations included
  (topic 0.849 → 0.842 / 0.834; lang 0.966 → 0.970 / 0.966).
- Retention_J − retention_random is within ±0.02 for all features.
- Even at r=4000 the excess drop beyond random is at most 8% of the above-chance margin (topic, nexttok,
  tense) and 1% for col/depth.
- Per-token k=25 gradient-pursuit removal (the paper's "non-J-space component") is likewise flat:
  ≤0.006 drop retrained, ≤0.03 frozen.
- Full curves over r ∈ {25, 200, 256, 384, 500, 1000, 1024, 1536, 2000, 4000}, retrained and frozen, each
  with an equal-rank random control, are in `probe_results/<f>.json → removal["43"]`.

**What the J-space paper's POS/tense evidence actually is.** The paper's evidence is the appendix
"Directed modulation affects the J-space more than other representations". For each of 4 categories
(coding language, verb tense, part of speech, capitalisation) with n=24 stimuli, they build a
**mean-difference probe from real positive vs negative stimuli and project out its J-space component, the
top k=25 J-lens directions of the probe vector itself by gradient pursuit**. They then show that an
"imagine" instruction moves this J-orthogonalised probe by ≤~1 SD, a real stimulus moves it 3-6 SD, and the
label word (e.g. `past`) stays far down the lens ranking.

- There is **no random-subspace control** and no retrained probe after removing a J-subspace from the
  activations. The J-orthogonalisation removes only 25 directions from one probe vector.
- On Qwen3.6-27B we reproduce the ingredients:
  - The J-orthogonalised mean-diff probe keeps its AUROC: tense 0.853 → 0.848, POS 0.908 → 0.904.
  - The labels are absent from the lens.
- But under the same test topic also survives (AUROC 0.900 → 0.880), as does language (0.941 → 0.943).
- Calibrated reading: probe survival after J-orthogonalisation is necessary but not sufficient for "outside
  the J-space", because strong features are redundantly encoded and survive any equal-rank removal. The
  paper's stronger evidence is the dissociation under the imagine instruction (J-lens moves, probe does
  not) plus label absence. We did not run the imagine-instruction test.
- On Qwen, the diff-in-means directions of tense and POS are clearly J-enriched (5.6x and 2.6x at r\*, vs
  2.2x for generic activations). Their top J atoms are temporal concepts (当时, 一年后, July, campaign) for
  Past−Present and adjectives (diverse, tropical, qualitative) for ADJ−NOUN. So the *value* is carried
  partly by J-space content (temporal setting; predicted adjectives) even though the *label word* never
  appears.

### G4: causal use (steer at L42, all context positions, 120 held-out items, 4 random unit directions of equal norm)

Behavioural metrics:

| feature | metric |
|---|---|
| col | log-odds that the next token contains "\n", at line ends and 1-2 tokens before |
| pred_pos | log P(adjective lexicon) − log P(noun lexicon), at positions the model predicts ADJ/NOUN |
| tense | log P(past verb forms) − log P(present verb forms), at verb slots |
| depth | expected number of `]` in the next token, at closing positions |
| lang | log P(Cyrillic tokens) − log P(Latin tokens) |
| col_report | expected first digit of the reported count |

Signed effect = (Δ(+d) − Δ(−d))/2:

| feature | full direction q90 / q99 | J-removed r\*=384 q99 | J-removed 4r\*=1536 q99 | max random q99 | KL (full / jrem1536 / random) q99 |
|---|---|---|---|---|---|
| col | 1.27 / 1.85 | 1.79 | 1.47 (79%) | 0.03 | 0.097 / 0.052 / 0.008 |
| depth | 0.093 / 0.171 | 0.154 | 0.070 (41%) | 0.047 | 0.22 / 0.15 / 0.02 |
| pred_pos | 0.30 / 0.54 | 0.32 | 0.18 (33%) | 0.016 | 0.059 / 0.022 / 0.015 |
| tense | 2.06 / 3.12 | 2.29 | 0.59 (19%) | 0.069 | 0.42 / 0.021 / 0.009 |
| lang | ≈0 / 0.17 | 0.14 | 0.008 (5%) | 0.014 | 0.13 / 0.028 / 0.017 |
| col_report | 0.82 / 1.11 | 1.07 | 1.01 (91%) | 0.35 | 5.1 / 4.4 / 0.15 |

- The effect retained after removing the 4r\* J-space from the steering vector separates the features:
  - count features: 79-91%
  - POS / tense / depth: 19-41%
  - language: 5%
- This retained fraction is a useful extra diagnostic: the causal route of tense and language runs mostly
  through J-space directions.
- The J-orthogonalised (k=25 atoms) direction keeps 85-95% of the effect for every feature, so it is too
  weak a removal to matter.
- col_report steering is disruptive at every dose (KL 2.7-5). Treat its effect as an upper bound.

### Leak pre-check (steer J-removed direction at J-layer 38 = hs[39]; read J-lens at 40/42/44)

Setup:
- Direction recomputed at hs[39] on 250 items, minus the top-r\*/4r\* J-space of M_38 (r\*(L38) = 192;
  we used L42's 384/1536).
- Dose = q99 and 2×q99 of the natural projection; 60 eval items; 2 random controls.

Results:
- With the J-removed directions, the target labels never enter the top-10 (POS/tense 0.000; col number
  tokens 0.5-1%, unchanged).
- The log10-rank shifts of the target labels (≤0.18) are within the range of the random directions
  (≤0.25), often with the wrong sign.
- Top-10 overlap with the unsteered readout stays 0.84-0.88, vs 0.80-0.82 for random directions.
- Depth is noisy: random directions move number tokens by ±1 log-rank.
- **No leak detected at these doses.** Note that the full (non-J-removed) tense direction pushes `past`
  *down* by 0.5 log-rank.

### Self-report (property, not a gate; 3 formats, chat, prefix truncated at the probed position)

| feature | free | forced choice (logits) | numeric (logits) |
|---|---|---|---|
| col | Spearman 0.42 [0.31, 0.52]; exact 4%; MAE 16.4 vs 15.7 predict-mean | 33% (majority 38%) | Spearman 0.29 |
| depth | Spearman 0.25; MAE 3.1 vs 1.66 predict-mean | 18% (majority 34%) | Spearman 0.56 |
| tense | 87% | 83% | AUROC 0.91 |
| next_pos (actual next word) | 52% | 47% (chance 25%) | AUROC 0.68 |
| topic | 82% | 87% | AUROC 0.999 |

- Counts are poorly reportable: col is at or below predict-mean in MAE.
- Tense and topic are reportable on request because they are computable from the text.
- Self-report was not run for pred_pos or lang.

### Replication of the paper's line-width experiment (Fig. 21) on Qwen (`s6`, 30 digit-free passages)

- Number tokens in the J-lens top-3 (J-layers 26-58) at passage positions: 0.50 (linewrap) vs 0.63
  (explicit report) vs 0.67 (first-letter) positions per passage, out of about 69.
- **The "pulled into J on demand" effect does not replicate on Qwen.** Qwen also cannot count without CoT:
  explicit report is exact on 7% and within ±2 on 53%; continuations deviate from the width by 7.6 chars.
- A first run on passages that contained digits gave about 7 positions in all conditions, all driven by
  digits copied from the text.
- This is why IJ1 (`col_report`) ends up non-J.

## J-lens choice and J-space definition

- **Lens files.**
  - `neuronpedia/jacobian-lens` `Qwen3.6-27B_jacobian_lens_n1000.pt` (1000 wikitext prompts, 63 source
    layers, fp16) matches the paper's definition (≈1000 pretraining-like prompts). It is WorkspaceBench's
    `jlens` artifact and our **primary** lens, now also in the shared HF cache on mats-weights.
  - `camilablank/workspace-lenses` `j-lens/lens.pt` is `n_prompts=25` (pile-10k).
  - The two differ substantially at L42: relative Frobenius difference 0.58; mean cosine of J-lens vectors
    0.76 (computed with gain w, not 1+w; minor); top-25 overlap 17% on random inputs. Use n1000.
- **J-space.**
  - Paper: a union of k-dim cones of nonnegative sparse combinations (k≈25, gradient pursuit) of J-lens
    vectors, the rows of W_U·J_ℓ. Ablation = per-token projection out of the top-10 active J-lens vectors.
  - Lead (after Viswanath): top-r right singular vectors of W_U·diag(g)·J.
  - We report both. The k=25 pursuit component explains 2-6% of ‖h‖² at L42, in line with the paper's
    "≤10%".
- **r\* (basic_readout, n=100).** L38: 192, L40: 384, L41: 384, **L42: 384**, L44: 256. Top-10 agreement at
  L42 by rank: 256 → 0.89, 384 → 0.93, 1024 → 0.95, 1536 → 0.97.

## Caveats

- **Gain bug, fixed.** Qwen3_5RMSNorm multiplies by (1 + weight). An early version of s3 used the gain
  `weight` alone, which affected the s3 J-space bases and the first jrem steering directions. Everything in
  this file was rerun with 1+w. s2, s8, s9 and s6 used the model's own norm module throughout.
- **Single dataset per feature; linear probes only.** The `col` count is only partly linear (R² 0.66).
- **Steering is teacher-forced logit metrics, not generation.** The `depth` metric is conditional on
  closing positions.
- **Lexicons are hand-built** for the tense, lang and topic metrics, and corpus-built (≥80% purity) for
  POS.
- **Old G2 random controls are single draws.** The random-subspace frozen-probe controls in the old-G2
  curves are one draw per r and non-monotonic. The 10-draw versions are in `jonly/`.
- **r\* rests on 100 items** at a coarse rank grid.
- **The functional criterion is very sensitive at n=40k.** Depth fails it with an AUROC of 0.503.
- **Steering and leak checks ran only at L42 (steer) and L38 (leak);** 40/44 sensitivity covers G1-G3 only.
- **topic and nexttok steering ran only with full and J-orthogonalised directions,** not J-removed (r)
  directions.
- **Language identity is trivially decodable from tokens** (bag of the last 4 tokens: 0.96).

## Files

Code is in `experiments/02_nonj_features/`; outputs are under `/results/02_nonj_features/`.

- Shared code: `nonj_common.py` (images, layer convention), `nonj_lib.py` (hooks, J-lens atoms, NN gradient
  pursuit), `features.py` (labels).
- Data: `s1_data.py`, `s1b_data.py` → `data/*.pkl`. `s2c_predpos.py` → `acts/prose/pred.npz`.
- `s2_capture.py` → `acts/<dataset>/hs{40..45}.npy` (fp16), `lens_hs{42,43}.npz`, `jsp_hs{42,43}.npz`.
- `s3_probe.py` → `probe_results/`, `probes/<feature>_hs<i>.pt`. Each probe file holds:
  - `steer_dir_unit` (diff-in-means/covariance, unit), `steer_dir_unit_jrem_r{256,384,1024,1536}`,
    `steer_dir_unit_jorth`
  - probe weights folded to raw h
  - dose percentiles and residual norms
  - layer-convention metadata
- G2 and readability analyses:
  - `s3b_sens.py` → `probe_results_sens/` (L40/L44)
  - `s3c_jonly.py` → `jonly/` (J-only probe, enrichment, INLP, relative retention)
  - `s3d_functional.py` → `functional/`
  - `s7_lensbag.py` → `lensbag/`
- Causal and leak checks: `s4_causal.py` → `causal/`; `s9_leak.py` → `leak/`.
- Self-report and replication: `s5_selfreport.py` → `selfreport/`; `s6_linewrap_replication.py` →
  `linewrap_replication.json`.
- Lens setup: `s8_rstar.py` → `rstar.json`, `jspace_basis_L{38,42}.pt`; `s0_setup.py` → lens download
  and comparison (`lens_inspect.json`).
- Summaries: `summarize.py` (local summary), `gates.json`.
