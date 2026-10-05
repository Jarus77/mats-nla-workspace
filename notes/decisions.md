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

## 2026-10-05 — D9: Fable round 3 (handoff #3), all accepted
1. **Correction to D8:** the ± contrast is NOT dilution-immune. For a feature-bearing item h·v ≠ 0, so |h+Δv| ≠ |h−Δv|. Readers that renormalise (OL `16000·unit(h)`, J-Lens cosine) can anti-track from norm asymmetry alone, which predicts the toy pattern exactly.
   - **Test first:** norm-preserving steer h' = (h+Δv)·‖h‖/‖h+Δv‖ for all readers. If OL's anti-tracking collapses, norm-preserving steering becomes mandatory everywhere, including the headline. If it survives, try steering along the INLP subspace or the diff-in-means direction.
   - Check whether the NLA's norm-matched injection also renormalises.
2. **No subtraction of a noisy null.** Report the paired win rate and the per-item shift against a random-direction null band (K directions, same norm, same ± construction), with effect size in null-SD units. Fraction-of-probe stays as the magnitude measure.
3. **Leak detector:** two-sided and calibrated to the empirical null band of random-direction J-Lens AUROCs (a fixed 0.05 is about the noise floor at n=240).
   - With no spectral elbow, no rank makes a direction "J-invisible". The removal rank only constructs the steering vector; **the leak check is what certifies a dose as non-J**. State this in the prereg.
4. **Final G2** (with the old survival test reported alongside):
   - **Primary, geometric:** enrichment of the **diff-in-means** direction ≤ 1.5 at r* and 4r*. INLP top-k is a supplement.
   - **Functional:** the full J-Lens readout doesn't track the feature on natural items (inside the random band).
   - **Labels:** non-J = both pass; **diffuse-but-J-readable** = enrichment ≤ 1.5 but J-Lens reads it (reported, excluded); in-J = enriched and readable.
   - The J-only-probe test (log-loss / d′) becomes secondary.
5. **J-space paper's POS/tense evidence:** check whether it's survival without a random-subspace control. If so, those features may be diffuse-but-J-readable. If NJ2/NJ3 fail, P1 fires honestly, and "the standard non-J test is uninformative" becomes a result.
6. **Templates:** mined on a held-out split, frozen before steering, same procedure for every reader. Select by natural-item discriminability (not steering response). Keep the fixed label template as a 4th column, and report fraction-of-probe template-averaged with its range.
7. **Order:** the norm-preserving toy re-run first, then feature results under the new G2, then decide whether (A) has features or P1 fires.

## 2026-10-06 — D10: feature gates (K1 partially passed)
Gated at L42, with r* = 384 from the readout-agreement rule (4r* = 1536), J-Lens = neuronpedia n1000. The camilablank lens was fit on only 25 prompts (cosine 0.76 vs neuronpedia), so it's rejected.

| Feature | Probe @L42 | Enrichment r*/4r* | J-Lens AUROC [null band] | J-removed steering | Label |
|---|---|---|---|---|---|
| **col** (characters since last newline) | R² 0.655 | 0.67 / 1.10 | 0.500 [0.499, 0.501] | +0.99 newline log-odds, 49× random; keeps 79% of the full effect | **NON-J** |
| col_report (explicit count question) | R² 0.88 | 0.59 / 0.86 | 0.500 | 3.5× | non-J (**the in-J calibration failed**) |
| bracket depth | R² 0.945 | 0.57 / 1.16 | 0.5035 (just above the band) | 1.5–1.9× | diffuse-but-J-readable (borderline) |
| predicted POS | acc 0.823 | 2.62 / 1.56 | 0.481 | 11× | J-enriched, fails |
| tense | acc 0.845 | 5.6 / 2.3 | 0.52 | 7× (19% of the full effect) | in-J |
| language | acc 0.966 | 3.3 / 1.7 | 0.62 | ≈0 | in-J |
| topic (reference) | acc 0.849 | 6.0 / 2.3 | 0.83 | — | in-J |

- **K1 status:** 1 non-J feature (target ≥2), with in-J calibration available from language and tense. By the prereg this means "run with 1 plus a backup". The backup agent is redirected to count-like features (indentation, words since sentence start, list index), plus capitalisation and coding language.
- **Methodological finding (a likely write-up section).** The standard "probe survives J-removal" test is uninformative: every feature keeps ≥98% probe performance, *topic and language included*. Retention minus random-equal-rank-removal retention is within ±0.02 for all features. The J-space paper's POS/tense evidence (diff-in-means probe, its own top-25 atoms projected out, z-scores) has no random-subspace control, and on Qwen the same test also "passes" topic. Under our calibrated gate, POS and tense are J-enriched.
- **col is a strong ground truth.** The model uses it (49× random under J-removed steering), it's not in J (enrichment ≈ 1, J-Lens at chance), it isn't self-reportable (free-answer Spearman 0.42, MAE no better than predicting the mean), and steering at L38 doesn't leak it into J-Lens.
- **col_report didn't move into J under explicit report** (the paper reports numbers entering J). Number tokens are near zero in J-Lens on digit-free passages, and Qwen can't count without chain of thought (7% exact).
- **Steering doses** (‖Δ‖/‖h‖ at q90/q99): col 0.11/0.16, tense 0.11/0.16, language 0.14/0.23. 2×q99 is void under the FVE guard.
- **Gaps:** the J-only probe used accuracy/R², not log-loss/d′; steering sensitivity at L40/44 not done. A gain bug (Qwen's final norm uses 1+weight) was found and fixed, and all gated numbers come from corrected reruns.

## 2026-10-06 — D11: anti-tracking explained (toy, n=40 pairs)
- **Norm-preserving steering is a no-op for OL and the NLA.** Both renormalise their input: OL injects `16000·unit(h)`, and the NLA injects `‖h_marker‖·unit(v)` at its block-1 output. Rescaling h+Δv leaves the direction unchanged; all statistics are identical to two decimals.
- **The real confound is unequal dilution.** For a feature-bearing item, h·v ≈ 1.2× the q99 dose, so the + and − sides carry the item's other content at different relative weights. Fix: **centred steering** h − (v·h)v ± Δv (equal norm on both sides), `capture.centred_steer_vectors`.
- **Results under centred steering** (J-removed language direction, q99):
  - **OL:** 0.45 → **0.49** (inside both null bands). OL's anti-tracking was a dilution artefact.
  - **NLA:** 0.62, **+8.3 SD** vs the J-complement band (+4.7 isotropic). It still tracks.
  - **J-Lens:** anti-tracks under every construction (linear readout), 4–7 SD outside the J-complement band. That is real leakage.
- **The leak detector, calibrated to the J-complement band (~[0.48, 0.52]), FIRES** for the J-removed language direction at L40/42/44 (negative, z −4.3 to −5.4). So by our own rule the toy's NLA tracking **cannot be claimed as a non-J read**. This confirms Fable's "no rank is J-invisible", and shows the detector is strict enough to matter. A fixed 0.55 threshold would have missed it.
- **Prereg revision 3 (pending):**
  - centred steering is primary (additive reported);
  - null bands with K ≥ 20 from both classes (isotropic and J-complement), reported as percentile and z, never subtracted;
  - the leak test is two-sided against the J-complement band;
  - for J-Lens use mean shift / AUROC (paired-win bands are degenerate for a linear readout).
- For col, the gate-stage J-Lens AUROC was 0.500 inside a very tight band, which is promising, but the leak check must be redone with centred steering and the calibrated band.

## 2026-10-06 — D12: Fable round 4 (handoff #4), all accepted
1. **Two co-headlines.**
   - **Part A, "What counts as non-J?":** the survival test is uninformative across 7 features (topic included); the calibrated gate works; on Qwen, POS and tense are J-enriched.
   - **Part B, "col, deep":** n=1 stated in the section's first sentence; generalisation treated as open.
   - Figure 1 = whichever comes out cleaner.
2. **col is doubly clean.** Qwen can't count without CoT (7% exact), so a Qwen-based decoder can't re-solve col by rebuilding the text and counting: tracking at leak-free doses is activation reading. The causal split is a finding in itself: col keeps 79% of its causal effect after J-removal, tense 19%.
3. **Backups reprioritised.**
   - Non-count features first: capitalisation state, coding language, in-quote state, and NEW gender/number agreement of an upcoming noun (FR/DE).
   - Two count features for within-family replication: indentation and words since sentence start.
4. **Calibration gap → (a).** In-J count control = list index *with numerals in context* ("Item 7:"), gated for IN-J. Language and tense stay for K3. (c) is rejected: the natural col arm measures counting, not verbalisation. col's primary candidate strings are qualitative ("near the end of a long line" vs "at the start of a line"), with numerals secondary.
5. **Framing of Part A:** a property of the test, not of their paper. Rank-r removal can't remove a redundant feature, and the same recipe passes topic (0.90 → 0.88). Three differences from the paper: model, lens (25-prompt fit, cosine 0.76), 25 atoms vs rank 384/1536. The claim is "the test as run can't support the conclusion", never "the conclusion is wrong". Offer the random-subspace control as a drop-in, and gate col under both lenses.
6. **pred_pos was a test-design artefact.** The lens emits tokens, not POS labels. Re-run the functional gate by classifying the top-10 tokens by POS. Generalised gate: "the lens readout contains the feature's value in the form the lens emits."
7. **col-specific validity hole (newline decision = J).** The leak set includes "\n"/EOL tokens; q90 is the primary dose; separate "long line" (count) from "newline coming" (decision) templates. Framing: the count is non-J, the decision it feeds is J, and a reader could read either.
8. **Pivot** only if K3 fails, or if col AND the in-J count control are flat on all readers (→ P1 with a strong Part A). col_report is reported as a mini-finding (explicit report didn't move the count into J; a model difference vs Claude).

## 2026-10-06 — D13: round-4 follow-ups (feature gates)
- **col is NON-J under both lenses.** camilablank: enrichment 0.82 / 1.41, label AUROC 0.5010 (null 0.498–0.502). n1000: 0.66 / 1.10, label AUROC 0.5002. col_report is NON-J under both; tense and language are IN-J under both. **This is the lens-robustness line for Part B.**
- **pred_pos is IN-J** with the generalised functional gate (top-10 lens tokens classified by POS): n1000 AUROC 0.66–0.69 vs null ~0.50; camilablank 0.60–0.65. Caveat: near-circular, because the label comes from the next-token distribution, which the L42 lens partly shows. So it's excluded either way, and the earlier "fails" label was a test-design artefact, as Fable predicted.
- **Lens agreement** recomputed with gain 1+w: J-lens vector cosine 0.79 (p10 0.72); top-25 overlap on random inputs only 19%. The lenses differ substantially, which supports reporting both.
- col remains the single non-J ground truth.

## 2026-10-06 — D14: re-solving decomposition on WorkspaceBench (H4 / panel 2)
Regex families, L42, n=400.
- **The sample-matched text-only baseline ties the readers.** Base Qwen in NLA register, no activation, k=1: 0.46 pooled. NLA@42 0.47 (McNemar p=0.75); OL@42 0.40 (p=0.09). Stronger text arms: Qwen with the wsbench prompt-only template 0.76; Gemini 3.8 Flash 0.75.
- **No hits only a reader gets.** NLA 0/187 hits are missed by every text sample; OL 4/159, all multilingual items credited via Chinese code-switching (a scoring-form artefact). Caveat: the union of 8 text samples is best-of-8, so the fair comparison is sample-matched (57% of NLA hits overlap with the k=1 text arm).
- **Re-solving from the readout:** a text model given ONLY the NLA's readout scores 0.57–0.60, above the NLA's own 0.47. Readouts carry more re-solvable content than they get credit for.
- **Same-layer:** NLA > OL@42 pooled, 187 vs 159 hits (p=0.002).
- **Text-verifiable precision** (Gemini judge, 100 readouts each): NLA 0.62 (36% contradicted, mostly misquotes of the prompt), OL 0.52 (35% contradicted, wrong entities; 28% unsupported), text-only Qwen 0.92. Judge agreement kimi-k3 vs Gemini: κ 0.72–0.77 on 40 claims. The prereg's human-labelled subset is still TODO.
- **Possible WorkspaceBench issue:** re-running the frozen prompt-only floor template gives 0.85/0.58/0.82/0.77 vs frozen 0.67/0.42/0.69/0.59. Unverified; the frozen run had empty cells, and thinking mode is a suspect. Verify before reporting, and if confirmed, notify the WorkspaceBench authors.
- **Implication:** on WorkspaceBench, regex success at L42 is fully explained by text, so the benchmark can't separate reading from re-solving. This supports panel 2 / P2, and it is NOT evidence that the readers don't read. It strengthens the case for col and text-fixed steering as the decisive test.

## 2026-10-06 — D15: col PILOT (60 declared-pilot items, EXCLUDED from main analysis)
- **Primary cell** (L42, centred, 4r* J-removed direction, q90): NLA AUROC 0.50–0.53 (paired win 0.57–0.70), OL 0.49–0.51. All z within ±1.5 of the K=20 isotropic and J-complement null bands, for every template family. Fraction of probe: NLA ≤ 0.12, OL ≤ 0.03 (H2 bar 0.25). **No reader tracks col beyond random directions.**
- **The decoders don't pick up even the RAW col direction,** while it moves the J-Lens newline score strongly (AUROC up to 0.84).
- **The prereg's H2 "CI excludes 0.5" test is too weak.** NLA numerals@r*: 0.54 [0.53, 0.57] excludes 0.5, but random directions move it as much (z +1.4/+1.8). The null band must be part of every primary test.
- **Leak detector with the newline set:**
  - **r* (384) leaks** at L40/42/44 (newline z +2.6 to +3.0);
  - **4r* (1536) is clean** at L40/42; at L44, 1024 still leaks.
  - Main-run non-J steering = **4r* at L42**.
- **FVE guard:** q90 passes; q99 fails on the − side, but random directions at that norm fail too, so the guard measures norm, not col. col's projection is right-skewed. q90 is primary; consider a q10-based − dose.
- **Free-text mentions of line length:** 0 in 4,080 NLA and 1,680 OL readouts. The J-Lens natural top-20 never contains newline or number tokens.
- **Power:**
  - NLA−OL ΔAUROC at q90 is +0.015 / +0.014 / +0.028 (count_qual / count_num / decision). At 240 items the minimum detectable difference is ~0.02.
  - **The null band, not n, limits power:** the spread between random directions doesn't shrink with n. Even at infinite n the NLA's z ≈ +0.6/+1.4/+0.6.
  - **H3 for col will likely fail.** State this explicitly at the freeze.
- **Gap (critical):** no in-J calibration at matched doses in this pilot, so a col null can't yet be read as "readers don't read non-J". The toy (language, D11) showed raw-direction tracking (NLA 0.91, OL 0.84) and J-removed NLA tracking +8 SD, but at different doses and items. Next: matched-dose calibration (language, tense, and the numeral list-index in-J count once gated) on the pilot split, same pipeline.
- **Natural arm (descriptive):** the ridge probe correlates 0.88 with col; NLA/OL |ρ| ≤ 0.16.
- Cost ~4.4 GPU-h (~$19).

## 2026-10-06 — D16: (C) planted intermediates: yield and first reader pass
- **K4 passes, but at large doses.** At L38 with the centred swap, 431/614 items are kept (348 at a fixed alpha=3); L34 keeps 212, L42 542; L24/28 fail. Keep criteria: the answer flips, unrelated-prompt KL ≤ 0.1, the relation is still decodable, and an X-vs-Y probe at L42 reads Y.
- **Dose caveat.** Kept doses have ‖Δ‖/‖h‖ median 0.49 (q10–q90 0.33–0.71), about 1.5–3× the natural X–Y class-mean distance. That is outside prereg §4.5's "natural spread". At alpha ≤ 1.5 there are still 109 items (≥ 60 passes). At alpha=1 only 9–14 flip. The model stays coherent (neutral KL median 0.015 at alpha 2). [DECIDE, Fable: primary = alpha ≤ 1.5 subset?]
- **All readers report the PLANTED bridge** (200 items): OL 1.00, NLA 0.98, J-Lens 1.00 for planted Y (vs 0.01–0.03 before planting), z ≫ null bands. This is in-J calibration (the swap direction is 6.5× enriched), not non-J evidence.
- **Confounds:**
  - by L42 the model already encodes Y's answer, so a reader could back-infer Y;
  - **the NLA rewrites the clue to fit the planted bridge** (12/12 free-form, e.g. "Eiffel Tower" → "Volvo… Sweden"). It follows the activation and confabulates a matching prompt. That's a qualitative H4-relevant finding: the NLA's "prompt reconstruction" is generated from the latent, not read from the text.
- **Dose contrast with col:** readers track in-J plants at ‖Δ‖/‖h‖ ≈ 0.49 but col was steered at 0.11 (q90). The matched-dose calibration (running) is what decides whether the col null is interpretable.
- **Not done:** FVE guard on plants, OL@40/44, the NLA subject-recovery check, and a J-removed-component plant with the leak check (the optional split arm).

## 2026-10-06 — D17: WorkspaceBench prompt-only floor check
- **Confirmed issue, cause inconclusive.** Under the thinking-off rendering every reader uses, the frozen mt prompt-only floors (0.67/0.42/0.69/0.59) are ~0.14 too low. Re-run: 0.83/0.47/0.84/0.80, significant on 3 of 4 families (z ≈ +2.2 to +3.3); an earlier independent draw gave the same pattern.
- No rendering reproduces the frozen values exactly: thinking-on with text kept and the raw no-template rendering are closest (~0.08). The frozen run's empty cells are unexplained; the generator (olens_sglang) isn't public.
- **Implication:** treat the frozen floors as a lower bound. Published reader-vs-floor gaps flatter the readers. Our conclusions rest on our own text arms, not the frozen floor.
- **Action [DECIDE, user]:** consider a polite, factual issue on camilablank/workspace-bench with the re-run table (an outward-facing action, so it's the user's call, ideally after the sprint or once the write-up is drafted).

## 2026-10-06 — D18: templates FROZEN
- `experiments/07_template_mining/templates_frozen.json`, sha256 09156722b059edb99d01e919c29b2ff660e37cba3152f2d3ce67f3272d3e923b. 14 features; mining and selection on fresh natural items; selection by natural-item discriminability only (no steering data). The evaluation must exclude all mining and selection ids listed in the JSON.
- **Natural-item selection AUROC @42** (mean of 3 mined templates; fixed label in brackets):

  | Feature | NLA | OL |
  |---|---|---|
  | col count | .60 [.51, .69] (.50) | .55 (.45) |
  | col break | .54 | .55 |
  | language | 1.00 | 1.00 |
  | tense | .85 | .92 |
  | **listnum, in-J count control ("Item N:")** | **.84** | **.99** |
  | indent | .84 | .89 |
  | quote | .83 | .76 |
  | capitalisation | .86 | .93 |
  | code | .99 | .99 |

- **The readers CAN talk about counts when the count is J-visible** (listnum: NLA .84, OL .99) but barely discriminate col (≤ .60) on natural items. That addresses the "can't verbalize counts" objection in the natural arm. Caveat: listnum's numeral is in the text, so its natural success may be text re-description; the steering arm is still needed.
- **Base rate:** 0/320 NLA and 0/960 OL readouts ever mention line length, column or wrapping. The col templates are style-matched paraphrases, not recurring phrasings.
- Mined templates mainly help OL (continuation forms beat label bullets). For the NLA the fixed label is often as good.
- Selected AUROCs carry a best-of-~7 winner's curse, so re-estimate on evaluation items.

## 2026-10-06 — D19: backup features: none qualify; the in-J count control failed
All 9 features were gated with 02's procedure plus D9/D11 (centred steering, two-sided J-complement leak bands), at 260 items each, at ~$80 (20 H100-h).

| Feature | Label |
|---|---|
| Capitalisation | **IN-J** (enrichment 2.71; J-Lens 0.81; 7× causal) |
| Coding language | **DIFFUSE-BUT-J-READABLE** (1.49/1.33; label 0.71; steer leaks z 2.4–5.8) |
| In-quote | **IN-J** (2.22/1.70) |
| French gender | **fails G1** (AUROC 0.60) |
| Indentation | **decodable, unused**: non-J by G2 (1.11/1.14), but G4 fails, i.e. steering doesn't move indentation (probably copied from the previous line's whitespace) |
| Words since sentence start | **DIFFUSE-BUT-J-READABLE** (lens shows rising sentence-end punctuation; leaks) |
| **List index with numerals ("Item N")** | **FAILED as the in-J count control**: digits never reach the J-Lens top-10, enrichment is no higher than shuffled-label, no steering effect |

- **Pattern (a finding for Part A):** on Qwen L42, count/positional features (col, indentation, sentence position, bracket depth) are J-depleted. Every non-count feature is J-enriched or J-Lens-readable, keeping only 12–23% of its steering effect after J-removal (col 79%). So **non-J content here looks like positional bookkeeping**. We can't defeat the "col is special because it's a count" objection with a natural non-count feature, so state it as a scope limit.
- **Numbers don't enter Qwen's J-space** in these settings (consistent with col_report). There's no in-J count control, so the readers' natural success on listnum (D18: NLA .84, OL .99) can't be attributed to J-space reading and may be text re-description. **The count-verbalisation calibration gap remains open.** Mitigations: qualitative templates (D12), plus language/tense for K3.
- **Warning on col's non-J label:**
  - Under the emitted-form functional test with the **full-distribution newline mass**, col reads weakly (Spearman 0.13, AUROC 0.58).
  - With the top-10 form it's inside the band (0.501).
  - Interpretation: the newline *decision* is in J (as Fable predicted), and it correlates with col. **Report col as "count non-J; the decision it feeds is weakly J-visible"**, and keep the count-vs-decision template split.
  - The primary functional rule stays label words + top-10 categories (one-sided, as in 02), with soft mass as a stricter supplement.
- **Method recommendation:** always report a shuffled-label enrichment reference (random relabellings already give 1.2–2.5× at r*).
- camilablank agrees with n1000 on every label except two (capitalisation DIFFUSE; indentation soft-mass above band).
