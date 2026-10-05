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
