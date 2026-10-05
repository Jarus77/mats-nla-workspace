# Decision log

## 2026-10-05 — D1: problem and design
- Problem: Neel's "Is NLA's extra really the non-workspace stuff?" on Qwen3.6-27B.
- Design: unified source attribution (user chose it over a narrower option).
- Overlap: a concurrent pilot (LittleMeHere/nla-vs-lenses). We differentiate on rigor and on natural non-J content, cite the pilot in paragraph 1, and don't copy blindly.

## 2026-10-05 — D2: revision after Fable critique (handoff #1)
Accepted:
1. **Attribution is a 2×2, not 4 buckets.** Content location (J / non-J) × reader mechanism (tracks activation / tracks text), plus "unsupported". Measured at population level by intervention sensitivity (move the feature with text fixed; move the text with the feature fixed). The old buckets weren't a partition: e.g. character count is non-J *and* deterministic from the text.
2. **Cut (B) split-injection.** Viswanath already has the 70B result, it's the pilot's next step, and it's the most out-of-distribution tool. Dose-response steering along natural probe directions is its on-distribution version. Optional 20-item replication if time allows.
3. **Headline = (A)**: 2–3 natural non-J features plus 2 in-J calibration features, under dose-response steering. Readers are reported as a fraction of what the probe recovers.
4. **Self-report is not a gate for non-J.** The criterion is surviving J-space removal plus absence from J-Lens. Phase 1 showed self-report failure is often an elicitation failure (colours are in J-space yet never named).
5. **(C) plants by moving the feature direction.** Not by adding J-Lens directions (trivial for J-Lens), and not by full-residual patching (that carries the counterfactual prompt along with it).
6. **Doses at p90, p99 and 2×p99** of the natural projection. Report the norm as a fraction of residual norm, plus next-token KL. Use the NLA reconstructor's FVE as an out-of-distribution detector: a dose is void if FVE leaves the pre-registered margin (new kill check).
7. **Scoring by paired log-odds**: candidates differ only in the feature value and are written in each reader's native format; AUROC is measured against a matched null. "Judged precision" is renamed **text-verifiable precision** (unsupported ≠ false).
8. **NLA re-solving ceiling**: a text-only LLM (TextNLA) given only the NLA's own explanation.
9. **Layer confound**: NLA at 42 vs Oracle Lens at 40/44, so report feature probe accuracy at 40/42/43/44.

Lead's caveats on the critique (sent back to Fable in handoff #2):
- **A linear probe is not a true ceiling.** Readers are nonlinear 27B decoders and could exceed it. Call it a reference, and show cases where a reader beats the probe as a finding, not an error.
- **On natural data, a text-tracking reader can get non-J features right** (character count is computable from the text). So results on natural data alone can't show "reads non-J". Only the steering arm can, and the write-up should lean on it.
- **The FVE out-of-distribution detector is partly circular.** The reconstructor was co-trained with the NLA verbalizer, so a high FVE on a steered activation shows the pair handles it, not that the NLA "reads" it. Pair it with a reconstructor-independent check: next-token KL, and whether the probe still decodes the intended value.
- **Paired log-odds for J-Lens** (no decoder): use the difference in J-Lens projection scores between the two candidate tokens. That's only comparable to the decoders after AUROC normalisation, so say so.

## 2026-10-05 — D3: novelty update (Viswanath follow-up) and J-Lens choice
- **Verified:** github.com/PranavViswanath/nla-introspection `jspace-followup/` (2026-07-13, unposted). On Llama-3.3-70B, natural safety-scenario contrasts are mostly non-J (enrichment 1.45, where 1.00 = random). With norm-matched components, the NLA gets full 0.55, J-only 0.18–0.38, non-J only 0.53–0.57, scored by keyword grading. There's no Oracle Lens and no Qwen. His own open caveat: "the verbalizer inferring the scenario from thematic residue rather than genuinely decoding" (the splice control is queued, not run).
- **Implication:** "the NLA reads natural non-J content" is partly shown on Llama, but **the reading-vs-re-solving question is explicitly open**, and that is exactly what our steering-with-text-fixed arm tests. Novelty claim, sharpened:
  1. Known-value natural features with independent ground truth, rather than thematic keywords.
  2. A causal test that separates reading from re-solving.
  3. All three readers, including Oracle Lens, on Qwen3.6-27B at matched layers.
- **Prior update:** the NLA reading natural non-J content is more likely than the prereg's 0.25. Fable to weigh in.
- **Candidate cheap secondary arm [DECIDE, Fable]:** Viswanath-style J-only / non-J-only projections on our features, for all three readers on Qwen. Inexpensive and directly comparable to his result, but watch scope.
- **J-Lens file:** the pilot and WorkspaceBench `jlens` both use neuronpedia/jacobian-lens n1000 (cosine readout). Our handoff named camilablank/workspace-lenses. Primary = neuronpedia n1000 (comparability); camilablank as a cross-check. The J-space rank r is set by a pre-registered rule, with curves over r reported (the spectrum is heavy-tailed with no elbow).

## 2026-10-05 — D4: Fable round 2 (handoff #2), all accepted
1. **Steering leak (a hole I missed).** Steering at L38 lets the model move non-J content into J-space before L40–44. Fixes:
   - steer along the **J-removed** probe direction;
   - **leak detector:** the J-Lens readout at 40–44 must not move, else that dose is void for non-J claims;
   - the **headline** steers at the reader's input layer; L38 steering is the on-distribution/behavioural validation.
2. **Probe as ceiling.** In the steering arm the probe is a true ceiling by construction (fraction > 1 is impossible there). "Reference, not ceiling" applies only to the natural arm, and only after H4 clears re-solving.
3. **FVE is a one-sided guard** (low ⇒ not reading; high ⇒ nothing). The real "doses in range" evidence is K3 in-J tracking at the same dose.
4. **J-Lens roles.** In the non-J arm it's the leak detector, not a competitor; in the in-J arm it's a competitor.
5. **FollowText clamp validation.** A probe retrained on clamped activations should be ≈ 0.5, else clamp a small subspace.
6. **H2 prior → 0.40. Co-headline = NLA vs Oracle Lens under text-fixed steering** (H3/H4 are the contribution; a bare H2 positive ≈ a Qwen replication of Viswanath).
7. **J-space rank.** G2 must pass at r* and 4r*. The write-up says "outside the top-r J-space", never "outside the workspace".
8. **G4 stays a gate,** ablating along the J-removed direction. Features failing it go to an appendix as "decodable, unused".
9. **Calibration features.**
   - IJ1 is gated like the others.
   - IJ2 → language identity (multilingual_mt) or typo target.
   - NJ3's ground truth = the model's *predicted* POS.
10. **Viswanath-style projection arm:** included, capped at half a day, appendix by default (norm-matched projections are themselves out-of-distribution).
11. **NLA checkpoint and K0.** NLA iter400 primary, 300 for robustness. For K0, functional tests outrank cosine. Fix the DeltaNet kernels first; if cosine stays ~0.974, proceed with the FVE-matching convention plus a sensitivity analysis on the headline arm.
12. **Figure 1 = dose-response** (three readers; one non-J and one in-J feature side by side). The 2×2 is an intro schematic. Add a ≤1-page descriptive re-analysis of the pilot, and cite the pilot and Viswanath in paragraph 1.

## 2026-10-05 — D5: K0 PASSED (NLA layer convention)
- The NLA was trained on the **output of decoder block 42 = HF `hidden_states[43]`**, which is the same as WorkspaceBench's "layer 42". No off-by-one.
  - Block-42 output: relative L2 0.026 (median 0.017), cosine 0.9999, best match on 64/64 examples.
  - Block-41 output: relative L2 0.22, cosine 0.974.
  - Functional test: FVE 0.760 on block-42 captures vs 0.695 on block-41.
- **Correction to handoff #2:** I reported the layer as "ambiguous, 0.974 vs 0.95". That came from reading the agent's partial output early. The 0.974 was the *wrong* convention. Tell Fable.
- **FVE:** iter400 0.761 (card 0.766), iter300 0.759. Controls:
  - av_base 0.459;
  - shuffled activation −0.90;
  - source text as explanation 0.15–0.21;
  - TextNLA 0.276.

  The reconstructor is off-style for these text-only controls, so the fair text-only ceiling is between 0.28 and ~0.46. A Sonnet-written control is deferred.
- **Out-of-distribution margin:** the SD within an example across k=4 samples is ~0.027, so compare steered vs natural FVE per example with k≥4. A shift of ~0.04 is detectable.
- **Oracle Lens contract:** trained on layers 20–60 in steps of 4, alpha 16000, layer number in the prompt. Run it at L40/44, never at 42.
- **Scoring gotcha:** paired log-odds must keep both candidates in one batch (batch composition shifts scores by up to 0.26 nats).

## 2026-10-05 — D6: WorkspaceBench replication PASSED
- All 12 published numbers (4 regex multi-token families × olens/nla/jlens) fall inside our 95% bootstrap CIs (n=100). Costs: OpenRouter $2.80, Modal ~1.5 GPU-h.
- **Matched-layer finding (new, worth a line in the write-up):** at L42 the NLA ties or beats Oracle Lens on all 4 families (multilingual 0.42 vs 0.28). Oracle Lens's published lead comes from the best-of-11-layers protocol, which is also a best-of-11-samples bonus.
- **Oracle Lens at L42 (off-contract) is no worse than at L40/44** (typo .76 vs .72/.70; basic .45 vs .42/.48). Proposal [DECIDE, Fable]: add OL@42 as the *same-layer* comparison next to OL@40/44. That removes most of the layer confound Fable flagged.
- Every arm is far below the prompt-only floor (0.69) on multihop_mt, consistent with heavy text inferability.
- Throughput: olens ~0.5 s/cell (H100), nla ~0.7 s/cell batched (H200, 119 GB), jlens ~0.17 s/cell (H200; it runs out of memory on an H100). Batched NLA generation differs from unbatched only in the last tokens (bf16).

## 2026-10-05 — D7: Fable on the same-layer comparison; best-of-layers checked mechanically
- **L42 is the primary layer for all three readers.** OL at 40/44 is sensitivity. That OL is no worse off-grid shows interpolation across its training grid is safe.
- **Mechanical check** (`experiments/01_wsbench_replication/best_of_layers_check.md`). The published rule is **per-item any-layer**: an item passes if *any* of the 11 layers hits. That is selection bias, and applying it reproduces the published values exactly. Gains over OL's per-family best single layer: typo +0.08 (0.78 @L24), multilingual +0.22 (0.43 @L56), multihop +0.16 (0.18 @L56), basic +0.11 (0.62 @L56).
- **Framing** (protocol asymmetry, not error): OL's layer flexibility is a real capability.
  - At OL's best single layer (mostly L56): typo 0.78 / multilingual 0.43 / multihop 0.18 / basic 0.62, vs NLA@42 0.73 / 0.42 / 0.15 / 0.57.
  - So **OL at its best layer slightly exceeds the NLA, within noise**.
  - **L42 is a weak layer for OL on multilingual/basic** (they peak late, at L56). The NLA's same-layer tie is therefore *less* impressive than it looks.
  - Report both, with the OL layer curve.
- **Remaining layer issues** (unchanged): the leak check for L38 steering, and feature probe accuracy reported at L42 specifically.

## 2026-10-05 — D8: pipeline smoke findings (language identity, n=40 pairs; plumbing, not science)
Pipeline built (`src/mats`, 45 tests pass). Smoke cost $1.54 GPU + $0.06 OpenRouter. Projected sprint compute: ~35 GPU-h, ~$150 including reruns.
Plumbing verified:
- capture matches `hidden_states[L+1]` (rel L2 ≤0.014);
- J-basis matches the feature agent's at overlap 1.0000;
- equal-norm random null 0.47–0.54 for all readers;
- FVE guard: q90/q99 pass (−0.015), 2×q99 void (−0.06);
- L38 arm: KL ≤0.008, 54–86% of the dose reaches 40–44.

Accepted changes (prereg revision 3, pending Fable):
1. **FollowFeature primary = paired / zero-dose-subtracted.** The cross-item AUROC is capped by each item's prior: with raw steering every reader moves on 100% of items, yet AUROC is 0.79–0.91. Report both. For zero-dose-subtracted contrasts, also subtract an equal-norm random-direction contrast to guard against norm dilution (cosine readout, OL unit-norm injection). The ± design is immune to dilution, but steered-vs-zero is not.
2. **Leak detector is two-sided:** void if |FollowFeature(J-Lens) − 0.5| > 0.05. Under J-removed steering the J-Lens *anti*-tracked (0.35–0.47), which the one-sided rule never caught.
3. **G2 is redefined (critical).** Language identity is 3–7× enriched in J-space, yet a probe survives removal of r* and 4r* at ≥0.998, and so does removal of an equal-rank *random* subspace. Strong features are redundant, so "survives J-removal" can't separate in-J from non-J. That would have silently mislabelled redundant in-J features (possibly POS/tense) as non-J. New rule, symmetric, using the random-subspace band:
   - **J-only probe:** trained on the projection *onto* the top-r J-space, compared with K=10 random equal-rank subspaces.
   - **Enrichment** of the probe and INLP directions.
   - **Non-J** = J-only probe inside the random band AND enrichment ≤1.5 (at r* and 4r*).
   - **In-J** = J-only probe above the band AND enrichment ≥2.

   The feature agent has been told.
4. **OL candidate templates must be mined** from OL's own generations (it writes content continuations, not label bullets).

Toy observation (NOT a result): on the J-removed language direction the NLA tracks (zero-subtracted 0.96, paired win 0.91, fraction 0.19), while OL (0.06–0.15) and the J-Lens (0.00) anti-track. This is the H3-type contrast. With a ± contrast it can't be norm dilution, so OL's anti-tracking needs explaining (the non-J part of the language direction may be anti-aligned with what OL uses).
