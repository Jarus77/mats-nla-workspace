# 03: pipeline smoke (toy in-J feature: French vs German)

**Verdict: the plumbing works end to end.** Every stage ran without error: capture, the J-space
basis, probes, J-removed directions, read-layer steering at 3 doses, three readers plus probe
and TextNLA, FollowFeature with clustered CIs, the leak check, the FVE guard and a mini arm U.
The total cost was **$1.54 of GPU (21.6 GPU-min wall) plus $0.056 of OpenRouter for 80 items**.

This validates plumbing, not science. n = 40 pairs, and the feature is trivially present in the text.

Date 2026-10-05. Library: `src/mats` (see `src/mats/README.md`). Layer L means block-L output,
i.e. `hidden_states[L+1]`.

- Readers:
  - NLA iter_000400 @42;
  - OL `olens_s3d_rl600` @42 primary (D7), 40/44 sensitivity, plus a natural layer sweep over 20..60/4 + 42;
  - J-Lens neuronpedia n1000 @40/42/44 (cosine readout);
  - probe = projection on the fold's own direction.
- Items: 40 content-matched FR/DE sentence pairs (`smoke_items.py`). Read position = last token.
  Probes use the last 4 tokens.
- Clustering: 2-fold cross-fitting by pair. Directions are fitted on one fold and applied to the
  other, so every item is held out. CIs = 2,000 pair-clustered bootstraps.
- Directions:
  - **raw** = class-mean difference;
  - **jrem** = raw with the top-r* J-space removed (r* from 02's `rstar.json`: 384 at L40/42, 256 at L44, 192 at L38).
- Doses: q90−m, q99−m, 2(q99−m) of the natural projection on the training fold. The resulting
  ‖Δu‖/‖h‖ is 0.05–0.28 (`metrics.json → checks.norm_ratio`).
- Null: equal-norm random direction at the q99 dose, drawn from the same subspace class (random-J for raw, random non-J for jrem).

## Plumbing checks

| check | result |
|---|---|
| Hook capture vs `hidden_states[L+1]` (batched + right-padded vs single sequence) | rel L2 0.010–0.014 at L40/42/44. This is bf16 batch numerics (00 saw 0.007–0.02); no off-by-one. |
| Our J-space basis vs 02's saved `jspace_basis_L{38,42}.pt`, at r* | subspace overlap **1.0000 / 1.0000** (faithful port) |
| Random-direction null, every reader, every layer | AUROC 0.47–0.54 (all CIs contain 0.5) |
| Probe on the steering contrast | 0.97–1.00, as it must be by construction |
| Page trend test, dose → LO shift | p < 1e-28 wherever a reader tracks. In the opposite direction (p = 1) for OL/J-Lens on jrem, which anti-track. |
| Template spread (3 templates, q99) | OL@42 raw 0.83/0.84/0.85; NLA raw 0.91 × 3; NLA jrem 0.53/0.62/0.63 |
| FVE guard (NLA, jrem, k=4, 24 items, paired, natural mean 0.70) | q90: −0.015 / −0.006; q99: −0.016 / −0.016 (pass). **2q99: −0.065 / −0.059, so VOID** (the margin rule fires as designed). 1 of 672 extractions failed. |
| Arm U (steer L38 at q99, read 40/42/44) | next-token KL 0.002 (jrem) / 0.008 (raw). The probe shift reaches the read layer: median 0.54–0.73 (jrem) and 0.67–0.86 (raw) of the injected Δ, i.e. ≥ 0.5, so it "propagated". |

## FollowFeature (arm R, mean of 3 templates; 95% pair-clustered CI)

| reader | dir | q90 | q99 | 2q99 | null q99 | fraction of probe @q99 | 0-dose-subtracted @q99 | paired win @q99 |
|---|---|---|---|---|---|---|---|---|
| NLA@42 | raw | 0.85 [.81,.89] | 0.91 [.87,.95] | 1.00 | 0.48 | 0.83 | 1.00 | 1.00 |
| OL@42 | raw | 0.79 [.76,.83] | 0.84 [.81,.88] | 0.99 | 0.50 | 0.69 | 1.00 | 1.00 |
| OL@40 | raw | 0.78 | 0.83 | 0.98 | 0.49 | 0.67 | 1.00 | 1.00 |
| OL@44 | raw | 0.74 | 0.79 | 0.99 | 0.53 | 0.60 | 1.00 | 1.00 |
| J-Lens@42 | raw | 0.80 | 0.84 | 1.00 | 0.50 | 0.69 | 1.00 | 1.00 |
| NLA@42 | jrem | 0.58 [.56,.60] | 0.60 [.58,.62] | 0.67 [.65,.70] | 0.50 | 0.19 | 0.96 [.92,.98] | 0.91 |
| OL@42 | jrem | 0.46 | 0.45 [.43,.46] | 0.44 | 0.49 | −0.10 | 0.15 | 0.19 |
| OL@40 / @44 | jrem | 0.45 / 0.43 | 0.44 / 0.41 | 0.42 / 0.39 | 0.50 / 0.49 | −0.12 / −0.18 | 0.08 / 0.06 | 0.12 / 0.09 |
| J-Lens@40/42/44 | jrem | 0.47 / 0.47 / 0.44 | 0.46 / 0.46 / 0.42 | 0.43 / 0.42 / 0.35 | ≈0.50 | <0 | 0.00 | 0.00 |

Natural (unsteered) AUROC by true label:
- probe G1, every reader, TextNLA-local, Gemini (prompt mode) and Gemini on the NLA's own
  explanation: all **1.00**;
- OL sweep: 0.99–1.00 at every layer from 20 to 60.

As expected, language identity is fully recoverable from the text, so the natural arm is
uninformative here. That is the point of the steering arm.

## Findings that should change the main-run design (for the lead / Fable)

1. **The across-item FollowFeature (prereg §2) is capped by item-level prior variance.**
   - For raw steering, every reader moves its LO on 100% of items (paired win 1.00, 0-dose-subtracted AUROC 1.00), yet the prereg AUROC is only 0.79–0.91 at q99.
   - The reason: a steered German sentence still reads "German" more than a steered French one.
   - **Recommendation:** pre-register the paired / 0-dose-subtracted AUROC (null (i) applied as a subtraction) as primary, or report both. `metrics.follow_feature(lo_zero=...)` already does this.
2. **The leak detector must be two-sided.**
   - Under jrem steering, the J-Lens moves *against* the feature on every item (paired win 0.00, AUROC 0.35–0.47).
   - The prereg rule (void if > 0.55) never fires on a negative shift, although the J-Lens readout clearly moved.
   - `analyze.py` also reports `moved_two_sided` = |AUROC − 0.5| > 0.05. With that rule, L44 jrem is flagged at every dose and L40/42 at 2q99.
   - Likely cause: removing the top-r J-space from w leaves a tail component that has a negative projection onto the label-token lens rows. The cosine readout also omits the norm gain g that the basis includes.
3. **Language identity fails the in-J G2 criterion.**
   - The J-space is enriched 3.2–7.0× over random (`checks.enrichment_w`).
   - But a frozen probe keeps AUROC ≥ 0.998 after removing r* *and* 4r*, as does a probe retrained on the J-removed vectors. The prereg's in-J requirement is ≤ 40% retained.
   - A binary feature this separable is linearly decodable from almost any subspace of a 5120-d residual. The equal-rank random-subspace control is also 1.00.
   - So IJ2 = language identity will not pass G2-in-J as written. Either change the in-J G2 rule (e.g. compare retention against the random-subspace control, or use enrichment / J-Lens presence G3), or switch IJ2 to the typo target.
4. **Readers diverge on the J-removed direction, even for this in-J feature.**
   - NLA: zero-subtracted 0.96, paired win 0.91, fraction of probe 0.19.
   - OL and the J-Lens: anti-track (0.06–0.15 / 0.00).
   - This is a toy with n = 40 and not a result. But it shows that the pipeline can produce exactly the NLA-vs-OL contrast H3 is about, and that OL behaves like the J-Lens there.
5. **OL templates need mining (§6.1 plan).**
   - OL's free generations at L42 are content continuations in the source language (often drifting into Chinese), not label bullets. Example: `- Le chat dort paisiblement dans son panier…`.
   - So "- French language" is off its output distribution. It still tracks (raw 0.84), but mined templates will likely be more sensitive.
   - NLA explanations open with the label directly ("French descriptive sentence…"), so NLA-register templates are on-distribution.
6. **The doses are already near the FVE edge.**
   - 2q99 (‖Δ‖/‖h‖ ≈ 0.15–0.18 for jrem) is void under the 0.05 margin.
   - q99 costs −0.016 FVE, about 0.6 within-example SD (0.027), so it is detectable.
   - Expect only q90/q99 to be valid for many features.
7. **Gemini returns no logprobs on OpenRouter.**
   - TextNLA-Gemini uses an elicited P(A), both option orders, LO = logit.
   - With `max_tokens` 60, hidden reasoning truncated 29/80 answers. Fixed with 600, now 0/208 unparsed (`n_unparsed` counter added).

## Compute and projected sprint budget

| stage | GPU | wall | $ (list) | per-unit rate |
|---|---|---|---|---|
| A: model + OL + J-Lens, all base-model work | H100 | 592 s (load 42 s, capture 48 s, OL scoring 332 s, OL sweep 71 s, TextNLA 36 s, J-Lens < 1 s, arm U 6 s) | $0.65 | OL paired LO **25.6 ms/pair** (12,960 pairs, 16 acts/forward); OL generation 1.23 s/row at batch 16; peak 65.7 GB |
| B: NLA AV + AR | H200 | 706 s (load 148 s, scoring 237 s, FVE 321 s) | $0.89 | NLA paired LO **55 ms/pair** (4,320 pairs); generate + reconstruct **0.48 s/sample** (672 at batch 64); peak 103.6 GB |
| Gemini TextNLA | — | 84 s | $0.056 | 208 calls, $0.00027/call |

- **Per item** (9 conditions × 2 directions × 3 templates; OL at 3 layers, NLA at 1; plus FVE on 24 items): about **16.2 GPU-s per item (1,298 s / 80) ≈ $0.019 per item.**
- Container start-up and image pull, roughly 1–2 min per container, are not included. The rates use Modal list prices of $3.95/h (H100) and $4.54/h (H200).

**Projection for the main run** (prereg §8: 5 features × 240 items; NJ1 has about 2.5 read positions per passage):

| block | basis | estimate |
|---|---|---|
| Arm R scoring | 9 conds × ~1.5 directions (jrem + 4r* robustness; raw for in-J) × 3 templates. NLA@42 0.055 s/pair; OL @42/40/44 0.026 s/pair | NLA ≈ 1.9 H200-h, OL ≈ 2.6 H100-h |
| Arm U | Forward passes are negligible; re-scoring the propagated captures costs the same as arm R | +1.9 H200-h, +2.6 H100-h |
| FVE guard | 60 items × 7 conds × k=4 per feature, arms R and U | ≈ 2.2 H200-h |
| FollowText + clamp validation, natural OL sweep, capture, probes and gates | | ≈ 1.5 H100-h |
| Free-form k=5 generations (§6.2) | 60 items × 3 conds × 5 per reader per feature | ≈ 0.6 H200-h + 0.7 H100-h |
| (C) planting + robustness (r sweep, NLA iter300, layer 43, templates) | about 2× arm R | ≈ 4 H200-h + 5 H100-h |
| **Total** | | **≈ 11 H200-h + 13 H100-h ≈ $100 at list.** With ×1.5 for reruns/debug: **≈ 35 GPU-h, ≈ $150.** |

- API: TextNLA-Gemini on the main run (5 × 240 × 2 modes × 2 orders × 3 templates ≈ 14k calls) ≈ $4. Judges per §6.2 are separate, roughly $10–20.
- **Cheap speed-ups**:
  - OL scoring at 32 rows/forward used only 66 of 80 GB, so raise the batch to about 128 rows.
  - Install `causal_conv1d`; transformers currently uses the torch fallback for capture/generation.
  - Score the NLA's 3 templates in one batch (6 candidates per activation) instead of 3 passes.
  - Each is likely about 1.5–2×.

## Files

- `smoke_items.py`: items, frozen candidate templates, J-Lens label words, Gemini questions.
- `env.py`: Modal image (infra pins + scipy, with `src/mats` shipped in).
- `smoke_modal.py`: app `mats-03-pipeline-smoke`. Stage A `stage_base` (H100), stage B `stage_nla` (H200).
  Run with `modal run smoke_modal.py [--stage base|nla|all]`.
- `gemini_textnla.py` (local) and `analyze.py` (local): produce `results/gemini_textnla.json`,
  `results/metrics.json` and `results/analyze_stdout.txt`.
- `results/stage_base.pt`, `results/stage_nla.pt` (git-ignored `*.pt`). These are also on the
  volume at `/results/03_pipeline_smoke/`, together with our J-space bases `jspace_basis_L{38,40,42,44}.pt`.

---

## Norm-preserving re-run (Fable round 3)

**Question.** Is OL/J-Lens anti-tracking on the J-removed direction a norm-asymmetry artefact?
When h·v ≠ 0, ‖h+Δv‖ ≠ ‖h−Δv‖, and readers that renormalise would shrink the item's existing
signal more on one side.

**Short answer.**
- **Rescaling to ‖h‖ changes nothing for any reader.**
- **An equal-norm ("centred") construction collapses OL's anti-tracking to chance** (0.49, inside the random-direction null band).
- **J-Lens anti-tracking survives everything.** It is a real linear projection of the J-removed direction onto the French/German J-Lens rows, i.e. a leak outside the top-r* J-space.
- **The NLA tracks under every construction.**

### How each reader consumes the vector (checked in code)

| reader | injection / readout | depends on ‖input‖? |
|---|---|---|
| NLA (`readers/nla.py` `_hook`) | at AV block-1 output: `h_marker + ‖h_marker‖ · v/‖v‖` | **No.** Norm-matched: only unit(v) matters. So the NLA renormalises too. |
| OL (`readers/olens.py` `_embed`) | the slot embedding is replaced by `16000 · v/‖v‖` | **No.** Only unit(v) matters. |
| J-Lens (cosine readout of record) | `(W_U J v)_t / ‖Jᵀ W_U[t]‖`; the denominator is per token, not ‖v‖ | **Linear in v.** No renormalisation; LO(h+Δv) − LO(h−Δv) = 2Δ·LO(v) exactly. |

Consequences:
- Fable's norm-preserving map h' = (h+Δv)·‖h‖/‖h+Δv‖ leaves unit(h+Δv) unchanged. **OL and NLA inputs are therefore identical** to the additive construction: measured max |ΔLO| is 0.05–0.24 nats, inside the 0.26-nat bf16 batch-noise band, and every statistic below is identical to two decimals.
- For the J-Lens the rescaling is a per-item factor of about 1 ± 0.02, so it barely matters.
- The rescaling *cannot* be the fix for OL. What can differ between the ± sides of a direction-only reader is **dilution**: when h·v ≠ 0 (median h·v ≈ 1.2× the q99 dose for jrem; ‖h+Δv‖/‖h−Δv‖ ≈ 1.02), unit(h+Δv) and unit(h−Δv) carry the item's other content at different weights.
- So we added a **centred** variant, h± = h − (v·h)v ± Δv. It has equal norms on both sides and removes this confound, at the cost of zeroing the item's own v-component.

### Setup

- Same 40 pairs, read-layer arm, L42 (J-Lens also at 40/44), 2-fold cross-fitting, q90/q99.
- Readers: OL and NLA use the template mean over the 3 frozen templates, with all 6 candidates of an activation in one forward batch; J-Lens uses label words.
- Directions:
  - raw (mean-diff);
  - jrem (mean-diff minus top-r* J-space);
  - inlp1_jrem (first INLP/logistic direction, J-removed);
  - inlp8_jrem (mean-diff projected onto the J-removed INLP-8 subspace).
- Null band: K=20 isotropic and K=20 J-complement random unit directions. They use the same q99 dose norm as the real direction (raw dose for raw, jrem dose for the J-removed ones) and the same ± construction and variant.
- Reported as percentile and z in null-SD units, never subtracted (`metrics.null_band_contrast`).
- Paired-win bands are degenerate for the linear J-Lens: each null shifts every item by the same constant, so the null win is 0 or 1. Use the mean-shift and AUROC bands there.

### q99 FollowFeature (cross-item AUROC [95% CI] / paired win / mean shift ± SD)

| reader | direction | additive (original) | norm-preserving | centred (equal-norm) |
|---|---|---|---|---|
| OL@42 | raw | 0.84 [.81,.88] / 1.00 / +6.15±1.87 | identical | 0.89 [.85,.93] / 1.00 / +5.91±2.78 |
| OL@42 | **jrem** | **0.45 [.43,.46] / 0.17 / −0.49±0.70** | **identical** | **0.49 [.47,.50] / 0.47 / −0.09±1.28** |
| OL@42 | inlp1_jrem | 0.48 / 0.15 / −0.17±0.23 | identical | 0.48 / 0.20 / −0.14±0.29 |
| OL@42 | inlp8_jrem | 0.46 / 0.20 / −0.41±0.60 | identical | 0.49 / 0.41 / −0.10±0.94 |
| NLA@42 | raw | 0.91 [.87,.95] / 1.00 / +13.0±6.5 | identical | 0.99 / 1.00 / +14.4±7.2 |
| NLA@42 | **jrem** | **0.60 [.58,.62] / 0.96 / +0.87±1.46** | **identical** | **0.62 [.60,.64] / 0.91 / +1.29±1.65** |
| NLA@42 | inlp1_jrem | 0.55 / 0.84 / +0.35±0.69 | identical | 0.56 / 0.90 / +0.45±0.80 |
| NLA@42 | inlp8_jrem | 0.59 / 0.96 / +0.74±1.24 | identical | 0.62 / 0.99 / +1.07±1.42 |
| J-Lens@42 | raw | 0.84 / 1.00 / +5.88±0.46 | 0.84 / 1.00 / +5.77 | 0.98 / 1.00 / +5.94 |
| J-Lens@42 | **jrem** | **0.46 / 0.00 / −0.38±0.08** | 0.45 / 0.00 / −0.42 | **0.46 / 0.00 / −0.36±0.04** |
| J-Lens@40 / @44 | jrem | 0.46 / 0.42 (win 0.00) | 0.46 / 0.41 | 0.46 / 0.42 |

q90 shows the same pattern: OL jrem 0.46 / 0.46 / 0.49; NLA jrem 0.58 / 0.58 / 0.59.

The centred raw-direction AUROC rises for every reader (OL 0.89, NLA 0.99, J-Lens 0.98). That is
expected: centring deletes each item's own u-projection, which carried most of the item-level
prior variance (finding 1 above).

### Null bands (q99; percentile among K=20 nulls / z in null-SD units; OUT = outside the two-sided 95% band)

| reader | direction | variant | vs J-complement nulls: mean shift | AUROC | vs isotropic nulls: mean shift | AUROC |
|---|---|---|---|---|---|---|
| OL@42 | jrem | additive = preserving | 0.00 / −4.6 OUT | 0.00 / −4.7 OUT | 0.05 / −2.1 OUT | 0.00 / −2.4 OUT |
| OL@42 | jrem | **centred** | **0.35 / −0.6 in** | **0.25 / −1.0 in** | **0.40 / −0.2 in** | **0.35 / −0.5 in** |
| OL@42 | inlp8_jrem | additive / centred | 0.00 / −3.8 OUT → 0.35 / −0.7 in | −3.7 OUT → −0.8 in | | |
| OL@42 | inlp1_jrem | additive | 0.10 / −1.4 in | 0.05 / −1.6 in | 0.30 / −0.6 in | in |
| NLA@42 | jrem | additive / centred | 1.00 / +10.6 / +15.8 OUT | +6.9 / +8.3 OUT | +5.4 / +7.9 OUT | +4.0 / +4.7 OUT |
| NLA@42 | inlp8_jrem | additive / centred | +9.0 / +13.1 OUT | +6.0 / +8.4 OUT | +4.6 / +6.5 OUT | +3.5 / +4.8 OUT |
| NLA@42 | inlp1_jrem | additive | +4.4 OUT | +3.2 OUT | +2.2 OUT | +1.8 OUT |
| J-Lens@42 | jrem | every variant | 0.00 / −6.1 to −7.0 OUT | −4.1 to −4.6 OUT | 0.0–0.1 / −1.7 to −2.0 (edge) | −1.6 to −1.8 in |

### Leak-detector calibration (step 3; J-Lens cross-item AUROC vs the 95% band of 20 random-direction J-Lens AUROCs, two-sided)

| layer | jrem AUROC (additive) | J-complement band | verdict | isotropic band | verdict |
|---|---|---|---|---|---|
| 40 | 0.46 | [0.49, 0.51] | **leak** (z −4.5) | [0.48, 0.53] | leak (z −2.0) |
| 42 | 0.46 | [0.48, 0.52] | **leak** (z −4.3) | [0.45, 0.54] | in band (z −1.7) |
| 44 | 0.42 | [0.48, 0.52] | **leak** (z −5.4) | [0.47, 0.55] | leak (z −3.2) |

- The J-complement band is very tight (±0.02): a random direction outside S_r* barely moves the J-Lens. Our J-removed feature direction is not "random non-J". It has a J-Lens-visible component in the spectrum tail beyond r*.
- Recommendation: calibrate the leak detector against the **J-complement null band, two-sided**, not against a fixed 0.55. Under this rule the toy's jrem direction leaks at every read layer, in the negative direction.

### What this means

1. **The prereg's norm-preserving fix is a no-op for OL and NLA.** Both readers see only unit(input), and the NLA norm-matches as well, so "renormalising readers" does not separate NLA from OL. Keeping `preserve_norm=True` as the default in `capture.py` is harmless; it only changes the J-Lens (slightly) and probes.
2. **OL's anti-tracking is a dilution artefact of additive steering when h·v ≠ 0.** With equal-norm (centred) steering it collapses to 0.49 [0.47, 0.50], paired win 0.47, inside both null bands (z −0.5 to −1.0). The same holds for inlp8_jrem (0.49, in band). **Under the fair construction OL does not track the J-removed direction at all.**
3. **The NLA's tracking is robust to the construction.** Centred jrem gives 0.62 [0.60, 0.64], paired win 0.91, +8.3 null-SD (J-complement band) and +4.7 (isotropic band). It is weaker but still out of band on the 1-D INLP direction (0.55, +3.2 SD).
   - So the NLA-vs-OL contrast survives, and gets sharper: OL goes from below chance to exactly chance.
   - It is still n=40 on an in-J toy feature where the J-removed direction leaks into the J-Lens (point 4). This is plumbing evidence, not a finding about non-J reading.
4. **The J-Lens anti-tracking is real, not an artefact.** It is linear, identical across constructions, and 4–7 SD outside the J-complement band. Removing the top-r* J-space does not remove what the J-Lens reads.
5. **Step 4 (INLP) changes nothing qualitatively.**
   - The INLP-8 J-removed direction behaves like the mean-diff jrem direction.
   - INLP-1 has a much smaller natural projection (h·v / dose ≈ 0.15 vs 1.22), so it has almost no dilution asymmetry (norm ratio 1.001). There OL is in band even additively (−1.4 to −1.6 SD), and the NLA tracks weakly but out of band (0.55).
   - This is consistent with the dilution account: the asymmetry scales with h·v.
6. **For the main run:**
   - (a) use centred/equal-norm steering (`capture.centred_steer_vectors`) or report it alongside additive;
   - (b) report every reader against K≥20 same-norm null directions from both the J-complement and isotropic classes, with no subtraction;
   - (c) leak detector = two-sided, J-complement null band.

### Cost and throughput (re-run)

- prep: 1 × H100, 76 s of work.
- OL: 3 × H100 chunks, 590 s scoring each, **76 ms per activation for 6 candidates** = 12.7 ms per candidate sequence at 96 rows per forward. That is no faster than at 32 rows, so OL scoring is compute-bound.
- NLA: 4 × H200 chunks, 810–840 s each, **142 ms per activation for 6 candidates** = 23.6 ms per sequence (about 15% faster than the smoke). Peak 76 GB without the AR.
- In total about 0.6 H100-h + 1.0 H200-h ≈ **$7** for 23,120 activations × 2 readers (289 conditions × 80 items).
- At these rates a K=20 × 2-class null band costs about 4× the real-direction scoring. Budget it only at q99 and only for the headline cells (about +4 H200-h / +2 H100-h on the main run).

Files: `normrerun_lib.py` (condition builder), `normrerun_modal.py` (app `mats-03-normrerun`:
`prep`, `score_ol`, `score_nla`), `normrerun_analyze.py`, and in `results/normrerun/`:
`metrics.json`, `analyze_stdout.txt`, `spec.pt`, `jlens.pt`, `{ol,nla}_chunk*.pt`.
