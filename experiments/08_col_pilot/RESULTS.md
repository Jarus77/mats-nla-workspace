# 08: col feasibility pilot — PILOT, EXCLUDED FROM MAIN ANALYSIS

> **This is a declared pilot split.** The 60 passages are listed in `pilot_split.json` (sha1 of
> the whitespace-normalised paragraph plus its 80-character head) and **must be dropped from every
> main-run item bank**. The pilot exists to estimate effect sizes for power, to check the FVE guard
> and the leak detector on `col`, and to find bugs. **Nothing here was used, or may be used, to
> select templates.** The templates were fixed in `templates.py` before any scoring, and 07's template
> mining never sees these steering responses.

Date 2026-10-05. Model Qwen3.6-27B. "L42" means the output of block 42, i.e. `hidden_states[43]`.
Cost: about 4.4 GPU-h (≈ $19), with 16 containers running in parallel; wall time ≈ 75 min.

## TL;DR

1. **No reader tracks col beyond what random directions do.** Setup: read-layer steering at L42,
   centred construction, the leak-free 4r\* direction, q90 dose.
   - NLA: AUROC 0.50–0.53, paired win 0.57–0.70.
   - OL: 0.49–0.51.
   - Every z against the K=20 same-norm null bands is within ±1.5, for every template family.
2. **The NLA does move slightly and consistently in the "+col" direction.** The clearest case is
   the numeral family on the r\* direction: AUROC 0.54 [0.53, 0.57], paired win 0.75. But random
   directions of the same norm move it by similar amounts: z +1.4 against the J-complement band and
   +1.8 against the isotropic band, both inside the 95% bands.
   - So a bare "CI excludes 0.5" test (prereg H2 as worded) would pass for the NLA on noise-level
     effects. **The null band is required.**
3. **Leak detector (the newline/EOL set is now included).** The leak verdicts decide which
   direction the main run can use:
   - The **r\* = 384 J-removed direction leaks at every read layer (40/42/44) and both doses.**
     J-Lens newline scores rise (z +2.6 to +3.0 against the J-complement band), and the J-Lens
     "long/end" tokens anti-track (z −2.0 to −2.2).
   - The **4r\* = 1536 direction is clean at L42 and L40** (EOL z +0.1/+0.2; templates within ±1).
     It still leaks EOL at L44, where 4r\* is only 1024.
   - **For non-J claims the main run must use 4r\* at L42.**
4. **FVE guard: q90 is valid for every direction** (ΔFVE −0.009 to −0.031).
   - q99 is void on the "−" side for raw and r\* (−0.058 / −0.060). A random direction of the same
     norm is void too (−0.045 to −0.062), so this is a norm effect and not col-specific.
   - The "−" side is worse because col's projection is right-skewed: m − (q90 − m) is already below
     the natural 1st percentile.
   - **q90 primary, q99 secondary**, as the lead specified.
5. **Mention base rate is zero.** In k=4 free generations, natural and steered, neither the NLA
   (4,080 readouts) nor OL (1,680) ever mentions line length, character count or line position.
   The J-Lens natural top-20 never contains a newline or number token (median best newline rank
   10,596).
6. **Power: the null band, not n, is the binding constraint.** The null-band SD is dominated by
   direction-to-direction variance, which more items cannot reduce. At n → ∞ the NLA's z at the
   leak-free q90 cell would be only +0.6 / +1.4 / +0.6 (count_qual / count_num / decision).
   The item-level NLA−OL ΔAUROC is small (+0.015 to +0.028) and well powered at 240 passages
   (minimum detectable effect ≈ 0.02), but it falls far below the prereg H3 bar of 0.05.

## Design (what was run)

| element | choice |
|---|---|
| Items | 60 fresh wikitext paragraphs (02's cleaner, `PILOT_SEED` 8008), hard-wrapped at widths {30, 40, 50, 60, 70} (12 each). They exclude every paragraph in 02's wrap/prose pickles and every 07 template-mining passage (prefix-matched). One read token per passage: a random in-line token (no newline in it), index ≥ 8, after at least one newline. ids are truncated at the read token. col ranges 6–60 (median ≈ 33). |
| Directions | 02's covariance direction for col (`col_hs{L+1}.pt`), raw, plus J-removed at r\*(L) and 4r\*(L): 384/1536 at L40/42 and 256/1024 at L44 (`rstar.json`). At L42 we used 02's exported jrem vectors (cosine to ours 0.99999). Enrichment of jrem at its own rank ≈ 0. |
| Doses | q − median of the natural projection on 02's wrap bank (40k col rows, not the pilot items). ‖Δ‖/median‖h‖ at L42: raw 0.108 / 0.159, r\* 0.106 / 0.157, 4r\* 0.081 / 0.123 (q90 / q99). The pilot items' projections match the bank (median 0.85 vs 0.47; q90 11.4 vs 9.9). |
| Steering | Arm R (read layer). **Centred** h − (v·h)v ± Δv is primary; additive is reported. Readers: OL and NLA at L42; OL also at 40/44; J-Lens at 40/42/44; the 02 ridge probe as reference. |
| Nulls | K = 20 unit directions per class at the same norm (each real direction's dose), same centred ± construction, never subtracted. Classes: isotropic; J-complement at r\* (for jrem1); J-complement at 4r\* (for jrem4). Full sets at L42. OL@40/44 got jrem1 nulls only; J-Lens got all of them at every layer. |
| Templates (fixed, `templates.py`) | Three families, all 10 candidates scored in one batch per activation. **count_qual (primary)**: "near the end of a long line" vs "at the start of a line"; "a long line" vs "a short line". **count_num (secondary)**: about 60 vs about 20 characters. **decision**: "a line break is coming up" vs "still far away"; "next token is a newline" vs "another word". Written in each reader's register (NLA after `<explanation>\n`; OL as `- ` bullets). J-Lens: long/longer vs short/shorter; end/ending vs start/beginning; sixty/seventy/eighty vs ten/fifteen/twenty. The J-Lens decision family = the EOL leak set. |
| Leak set | J-Lens max cosine over every newline-containing token. In Qwen's vocabulary all 200 such tokens are whitespace-only, so "narrow" = "broad". Plus the count templates. Two-sided test against the J-complement band, on both mean shift and AUROC. |
| FVE | NLA AV + AR, k=4, per-example paired natural vs steered, margin 0.05 / natural p5 (`metrics.fve_guard`). 17 conditions, including 4 same-norm null directions. |
| Free text | NLA explanations from the FVE runs (17 conditions × 60 × 4). OL `verbalize` k=4 at 7 conditions. J-Lens natural top-20. |

## Results

### FollowFeature at L42, centred (template-family mean)

Columns: AUROC [95% CI], paired win, mean shift ± SD. The z columns give z against the null bands
in null-SD units (isotropic / J-complement); * = outside the two-sided 95% band. Full tables,
including L40/L44, the probe and per-template results, are in `results/analyze_stdout.txt`.

**Primary cell: 4r\* direction (leak-free at L42), q90**

| reader | count_qual (primary) | count_num | decision |
|---|---|---|---|
| NLA@42 | 0.50 [.48,.53] / 0.57 / +0.05±0.40; z +0.1 / +0.6 | 0.53 [.51,.55] / 0.70 / +0.12±0.44; z +0.8 / +1.0 | 0.52 [.50,.53] / 0.62 / +0.05±0.17; z +0.8 / +0.6 |
| OL@42 | 0.49 [.47,.51] / 0.37 / −0.10±0.58; z −0.4 / −0.6 | 0.51 [.49,.54] / 0.58 / +0.08±0.28; z +0.9 / +1.2 | 0.49 [.47,.50] / 0.42 / −0.08±0.32; z −0.7 / −0.3 |
| J-Lens@42 | 0.48 / 0.08 / −0.03±0.03; z −0.1 / −0.9 | 0.49 / 0.22; z −0.2 / −0.5 | EOL 0.50; z −0.4 / +0.1 (no leak) |
| probe (02 ridge) | 1.00 (fraction-of-probe denominator) | | |

**Other cells (not valid for non-J claims: r\* leaks; raw is in J by construction)**

| reader | dir, dose | count_qual | count_num | decision |
|---|---|---|---|---|
| NLA@42 | r\*, q90 | 0.52 / 0.67; z +0.5 / +0.8 | **0.54 [.53,.57] / 0.75; z +1.8\* (AUROC, iso) / +1.4** | 0.51 / 0.62 |
| NLA@42 | r\*, q99 | 0.54 [.51,.58] / 0.67; z +0.6 / +0.9 | 0.56 [.53,.59] / 0.72; z +1.5 / +1.4 | 0.52 / 0.70 |
| NLA@42 | raw, q99 | 0.48 / 0.42 | 0.54 / 0.62 | 0.53 / 0.65 |
| OL@42 | r\*, q90 / q99 | 0.49 / 0.50 | 0.50 / 0.50 | 0.49 / 0.48 |
| OL@42 | raw, q99 | 0.46 / 0.35 | 0.48 / 0.45 | 0.50 / 0.55 |
| OL@40, OL@44 | all | 0.46–0.51 | 0.49–0.52 | 0.49–0.53 (raw paired win up to 0.68) |

- **Fraction of probe** (probe AUROC = 1.00 at every cell): NLA ≤ 0.12, OL ≤ 0.03. The prereg H2
  hard bar is 0.25.
- **Additive vs centred** is qualitatively identical (NLA r\* q90: 0.54 additive vs 0.52 centred).
  OL shows no anti-tracking artefact here.
- **Per-template spread is small.** NLA r\* AUROC 0.51–0.56 across its five templates; OL
  0.49–0.52.
- **The raw direction moves neither decoder**, although its J-Lens EOL readout shifts strongly
  (AUROC 0.73–0.84, z +3.5 to +4.6 against the isotropic band). This is a reader-insensitivity
  result: OL and the NLA do not even pick up the raw col direction at in-range doses. For contrast,
  in the 03 language toy at a similar ‖Δ‖/‖h‖ ≈ 0.15 the NLA gave 0.99 on the raw direction.
  **This pilot has no in-J calibration feature at matched doses (K3).** The main run needs one
  before reading col nulls as "workspace-bound".

### Leak test (J-Lens, centred, two-sided against the J-complement band)

| layer | r\* q90 / q99 | 4r\* q90 / q99 | raw (reference; isotropic band) |
|---|---|---|---|
| 40 | **LEAK**: EOL z +2.7\*/+2.6\*; count_qual z −2.0\* | clean (EOL −0.3; count_qual −2.0, inside the band) | EOL +4.4\* |
| 42 | **LEAK**: EOL +2.8\*/+2.6\*; count_qual −2.2\* | **clean** (EOL +0.1/+0.2; count_qual −0.9) | EOL +4.6\* |
| 44 | **LEAK**: EOL +3.0\*/+2.9\* | **LEAK**: EOL +2.6\*/+2.5\* (4r\* = 1024 at L44) | EOL +3.6\* |

- The J-removed direction at r\* still carries the newline decision. The EOL J-Lens score
  rises with +col: AUROC 0.54–0.62, against a J-complement band of about ±0.01.
- Per the Fable round-4 rule, **any r\* tracking at any dose is void for non-J claims**. This
  includes the NLA's numeral-family movement above.
- Removing 1536 dimensions at L42 cleans the leak at both doses.
- **Recommendation:** in the main run, use 4r\* at L42 as the non-J steering direction. Either
  pre-register the L44 leak (OL@44 sensitivity cells are then void), or use r=1536 at L44 too.

### FVE guard (NLA, natural mean FVE 0.717, k=4, n=60 paired)

| condition | + side ΔFVE | − side ΔFVE | verdict |
|---|---|---|---|
| r\* q90 | −0.009 | −0.024 | valid |
| r\* q99 | −0.037 | **−0.060** | − side void |
| 4r\* q90 | −0.010 | −0.018 | valid |
| 4r\* q99 | −0.013 | −0.041 | valid (− side near the margin) |
| raw q90 | −0.010 | −0.031 | valid |
| raw q99 | −0.032 | **−0.058** | − side void |
| null isotropic #0 at the r\* q99 norm | **−0.060** | −0.045 | + side void |
| null J-complement #0 at the r\* q99 norm | −0.044 | **−0.062** | − side void |

- **The guard measures norm more than col.** Random directions at the q99 norm (≈ 0.157‖h‖) lose
  as much FVE as col does.
- The "−" side is systematically worse for col because the centred −Δ puts the projection at −9.3.
  The natural 1st percentile is −8.0 and the median +0.5, because the projection is right-skewed.
- **Main-run suggestion:** keep q90 primary. Consider dosing the "−" side to the natural lower
  quantile (q10 − m) instead of mirroring.
- Extraction failures: 1 of 4,080.

### Free generations: does any reader mention line length? (base rate)

- **NLA:** 0/240 natural readouts mention line length, characters or line position (strict regex).
  Across all 17 conditions there are 8 strict-regex hits in 4,080 readouts, and none is about line
  length. Five are "Toy Story 3 characters" (the numeral pattern firing). Two describe a visible
  "\n" inside quoted text ("abruptly breaking with a line break after 'charts'"): they mention the
  line break, not line length. One is from OL ("over 30 characters from the novel").
- **OL:** 0/240 natural; 1/1,440 steered (a false positive).
- **J-Lens@42 top-20:** no newline or number tokens on any natural item.
- **Bug found and fixed.** The first regex counted "character(s)", which matched fictional
  characters in 13% of NLA readouts. We fixed it after looking only at natural readouts and before
  any steering comparison; both rates are in the JSON.
- **Implication:** for col, free-text judged scoring (§6.2) would have a floor of 0. Teacher-forced
  paired log-odds is the only usable readout.

### Natural arm (descriptive only; n=60, |ρ| > 0.25 ≈ p < .05)

| reader | Spearman(zero-dose score, col) |
|---|---|
| 02 ridge probe @42 | **0.88** |
| NLA@42 | count_qual −0.09, count_num +0.16, decision −0.13 |
| OL@42 | −0.01 / −0.01 / −0.06 |
| J-Lens@42 | count_qual **−0.33** (anti-correlated), count_num −0.02, EOL +0.20 |
| TextNLA (base Qwen, prompt only) | count_qual +0.27, count_num −0.13, decision −0.09 |

- No decoder reads col from natural activations with these templates.
- **Qwen cannot count without chain of thought** (02: 7% exact). A Qwen-based decoder therefore
  cannot re-solve col by reconstructing the text and counting. Tracking at valid, leak-free doses
  would be activation reading.
- TextNLA is kept as a descriptive check only. Its weak +0.27 on count_qual likely reflects coarse
  cues such as distance into the passage.

### Power estimate for the main run

**Item-level contrast** (NLA@42 − OL@42, centred, item bootstrap, 80% power, two-sided α = .05)

| cell | ΔAUROC (SE @ n=60) | n for observed Δ | n for Δ = 0.05 | Δ paired win (SE) |
|---|---|---|---|---|
| 4r\* q90 count_qual | +0.015 (0.014) | ≈ 420 | 36 | +0.20 (0.08) |
| 4r\* q90 count_num | +0.014 (0.018) | ≈ 800 | 59 | +0.12 (0.10) |
| 4r\* q90 decision | +0.028 (0.012) | ≈ 90 | 27 | +0.20 (0.09) |
| 4r\* q99 count_qual | +0.025 (0.020) | ≈ 290 | 73 | +0.10 (0.09) |
| r\* q90 count_num (leaks) | +0.042 (0.017) | ≈ 75 | 51 | +0.35 (0.08) |

- At the planned 240 passages, SE scales by √(60/240), so the minimum detectable ΔAUROC is
  ≈ 0.017–0.025. The prereg's 0.05 bar is detectable from about 60 passages.
- The observed leak-free differences (≤ 0.03) are below that bar. **H3 as written is likely to
  fail for col**, unless the main run's mined templates are much more sensitive.
- Single reader against 0.5 (leak-free q90): the NLA needs n ≈ 3,000 (count_qual, AUROC 0.50) or
  ≈ 90 (count_num, 0.53). OL never departs from 0.5.

**The binding constraint is the null band.** `results/null_floor.txt` splits the null-mean SD
into direction-to-direction variance and item noise:

| NLA, q90, J-complement band | real shift | null SD = √(direction² + item-SE²) | z at n=60 | z at n → ∞ |
|---|---|---|---|---|
| count_qual, 4r\* | +0.051 | √(0.130² + 0.056²) = 0.141 | +0.56 | +0.61 |
| count_num, 4r\* | +0.118 | √(0.076² + 0.078²) = 0.109 | +0.99 | +1.41 |
| decision, 4r\* | +0.050 | √(0.076² + 0.020²) = 0.079 | +0.60 | +0.62 |
| count_num, r\* (isotropic band) | +0.270 | √(0.073² + 0.102²) = 0.126 | +1.79 | +3.08 |

- Random same-norm directions genuinely move the decoders' log-odds (they are perturbation-sensitive).
  That spread does not shrink with more items.
- **No main-run n makes the leak-free col effect leave the null band at q90.** Only a larger true
  effect could, from better templates or a higher valid dose.
- Larger K tightens the band edges but not the SD.
- **Recommendation for the prereg freeze:** state explicitly that the null-band z is
  direction-limited, and that the item-level CI against 0.5 is necessary but not sufficient.

## Bugs and design issues found (for the main run)

1. **r\* is not enough for col.** The r\* J-removed direction leaks the newline decision into the
   J-Lens at every layer. Use 4r\* at L42 (clean). At L44 the 4r\* = 1024 direction also leaks.
2. **The symmetric ± dose leaves the natural range on the "−" side** (skewed projection), which is
   what voids FVE there. Consider an asymmetric "−" dose at q10 − m.
3. **The FVE guard voids any direction at the q99 norm**, random ones included. It is a norm guard;
   treat q99 as secondary.
4. **The mention regex must not count "character"** (fictional characters). Fixed (`MENTION_RE_STRICT`).
5. **There is no in-J calibration at matched doses in this pilot.** Without K3 the col nulls cannot
   be read as "workspace-bound". The main run must run language/tense at the same ‖Δ‖/‖h‖.
6. **Qwen's newline tokens are all whitespace-only** (200 tokens), so the narrow and broad EOL leak
   sets coincide.
7. **Not piloted:** the prereg's multi-position design (≤ 4 positions per passage, ICC / design
   effect). Here there is one position per passage, so the clustering assumptions in §8 remain
   unchecked.
8. **Throughput** (10 candidates per activation, 100 rows per forward):
   - OL ≈ 0.12 s per activation (the H100 request was served on H200);
   - NLA ≈ 0.23 s per activation;
   - NLA verbalise + reconstruct ≈ 0.6 s per sample.

   The OL L40/44 and full-null L42 sets dominate the cost.

## Files

- `pilot_split.json`: the excluded pilot passages (**main run must drop them**).
- `templates.py`: the frozen fixed templates and mention regexes.
- `pilot_items.py`: the item generator.
- `pilot_lib.py`: conditions, centred/additive construction and the EOL sets.
- `env.py`: the Modal image.
- `pilot_modal.py`: app `mats-08-col-pilot` (`build_items`, `prep`, `score_ol` ×5, `score_nla` ×6,
  `fve` ×4, `gen_ol`). Run with `modal run pilot_modal.py`.
- `analyze.py`: produces `results/metrics.json` and `results/analyze_stdout.txt`.
- `results/null_floor.py`: produces `results/null_floor.txt` (direction-limited null bands).
- Raw outputs: `results/08_col_pilot/*.pt`, also on volume `mats-results:/08_col_pilot/`. Plus
  `results/run.log` and `run_summary.json`.
