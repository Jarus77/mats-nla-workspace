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

## 2026-10-06 — D20: K3 PASSES (matched-dose calibration, pilot split, excluded)
Language (ru vs pl; 02's direction) and tense, 60 items each, at L42, centred raw direction, q90.

| Reader | Language | Tense | Raw col at the same dose |
|---|---|---|---|
| NLA | **0.83** (z +25/+31) | **0.83** (z +12/+18) | 0.48 |
| OL | **0.76** (z +11/+21) | 0.55 (out of band, below the 0.65 bar) | 0.48 |

- In-J features were steered at slightly **smaller** relative norm (0.095) than col (0.108).
- **At equal relative norm** (0.11 / 0.14): NLA language 0.87/0.92, tense 0.85/0.89; OL language 0.80/0.85; **col 0.47–0.48 for both readers**.
- **The null band is not the limiter for in-J features:** band SD 0.012–0.021 vs col 0.022–0.029; in-J per-item d_z +1.2 to +1.5 vs col −0.2 to −0.3.
- **The col null is interpretable:** "these decoders don't read col", with the template caveat. Fixed-template sensitivity is large (OL tense 0.39–0.87 across templates), so the main run uses the frozen mined templates (D18) and adds OL continuation-style col candidates (newline vs another word).
- **New (pilot, secondary): the NLA tracks the J-removed (4r*) tense direction leak-free** (0.60, z +8.7, clean at L42/L40 under label and content leak sets), while OL is inside the band. So the NLA's failure on col is **feature-specific**, not a general inability to read outside the top-1536 J-space. Pre-register it as a secondary cell. The language 4r* result is void (Cyrillic content leak, z +3.7).
- **Leak detector upgrade:** every leak check includes a **content token set** besides label words (the label-only set missed the language leak).
- **Smaller caveat:** col's direction is a regression-covariance direction, while language/tense use class-mean differences. Add a binary short-vs-long col direction (class-mean) as a robustness arm.
- Pilot total ~9.7 GPU-h (~$42).

## 2026-10-06 — D21: Fable round 5 (pre-freeze), accepted
**Reframe (the data supports a better sentence).** The RAW col direction moves J-Lens newline to AUROC 0.84, yet neither decoder tracks it. The decoders fail on col even when its J-readable consequence is present, so **content type, not location**, explains the null. What the NLA reads beyond the top-J space is the distributed tail of semantic features (tense 4r*).

Working one-sentence answer (finalised after (e) is resolved):
> "On Qwen3.6-27B at L42, the content outside the J-space that the model actually uses is positional bookkeeping, and neither the NLA nor Oracle Lens reads it in any phrasing, even when its downstream decision is lens-readable; what the NLA reads beyond J-Lens is the distributed tail of features the model can already verbalise, and on WorkspaceBench its successes are matched by a text-only model."

The title's dichotomy is answered **"neither"**: the NLA doesn't read what the model can't say, and it doesn't re-solve; it *generates*. Decide the title after (e).

**Corrections:**
- **(e) possible inconsistency:** the re-solver saw ONLY the reader's readout (no prompt) and scores 0.57–0.60 > text-only k=1 (0.46), but < prompt-based arms (0.75–0.76). Per item: does readout-only re-solving ever succeed where ALL prompt-based text arms fail? Until that's resolved, don't freeze "fully explained by text".
- **(C) relabelled** as in-J calibration (not a re-solving test). Its real contribution is the 12/12 rewrite: the NLA's "prompt reconstruction" is generated from the latent, which kills "decode the text, then compute" as the mechanism and replaces it with "the decoder's LM confabulates a consistent story".
- **OL's col null is less secure than the NLA's.** OL passes K3 only on language (tense 0.55) and is template-fragile (0.39–0.87), so present the two decoders asymmetrically.
- **Natural col NLA 0.60 needs TextNLA on the same items** (expect text: long vs short lines are visible without counting).
- **Behavioural validation at q90:** report the realised causal (newline log-odds) shift at the exact dose and direction where the null is claimed.
- **Scope "non-J on Qwen = positional bookkeeping"** as a survey of ~12 features, not a theorem.

**Structure: four parts.**
- A: what counts as non-J, plus the survey.
- B: the col null with matched-dose positives. **Figure 1** = three readers × {language, tense, col} at matched relative norm with both null bands.
- C: what the NLA reads beyond top-J (the tense tail; the latent-generated reconstruction).
- D: WorkspaceBench = text (after (e)).

Tense 4r* goes in the main text as a pre-registered secondary, with its provenance in one clause. It's a geometric claim (tense is reportable at 87%), not "reads what the model can't say". If it doesn't replicate, it moves to the appendix.

**Count gap:** scope B to "in any phrasing they use, incl. newline-continuation phrasing". **New cheap cell:** next-token digit in a numbered list ("next digit 5 vs 7"), which is maximally J and should pass G4. If the readers track it, the col null is about location; if not, B is scoped to content type. Either way it resolves the gap.

**Main-run changes:**
1. **Significance threshold:** |z| ≥ 3 vs both bands.
2. **Equivalence test for col** with a declared SESOI (fraction of probe 0.25 ↔ a named AUROC).
3. **Content token sets** in every leak check.
4. **Templates:** mined, plus OL continuation candidates. **Promote the raw-direction cell with newline-continuation candidates to primary** (the count-vs-decision separation).
5. **col direction:** add the class-mean binary direction.
6. **Pre-register tense 4r* NLA-vs-OL** as a secondary.
7. **Planting:** alpha ≤ 1.5 primary, relabelled as calibration.
8. **n:** 240 col items split across the two col directions (a paired test with d_z ≈ −0.3 needs ~120). Redirect compute to the raw-direction newline cell, the next-digit cell, tense 4r*, and more OL tense items.

**Prereg framing:** a confirmatory replication of a pilot null + two positive controls + one secondary, with hypotheses in that order.

**WorkspaceBench floors:** file the issue first (re-run table, thinking-off detail), then one paragraph: confirmed, cause unknown, authors notified. The issue is outward-facing, so it needs the user's approval.

## 2026-10-06 — D22: (e) resolved; "text-recoverable" holds; the deck is built
- **The apparent inconsistency was an instruction mismatch.** The 0.46 text arm used the NLA's own instruction, which never asks for the answer; the re-solver instruction asks for the entity plus the language, which is what the regex rewards.
- **Fair same-template, k=1 comparison** (Gemini, n=400):

  | Re-solver input | Pooled |
  |---|---|
  | Prompt only | **0.89** |
  | Readout only, NLA / OL | 0.60 / 0.56 |
  | Prompt + readout | 0.88 / 0.88 (adding the readout doesn't help; p=0.38/0.47) |

  Readout-only hits that every prompt arm misses: 4 NLA, 2 OL of 400. All are regex-form artefacts (uncredited short forms, OL's Chinese code-switching copied by the re-solver, one name-order variant).
- **Defensible wording (Part D):**
  > "On WorkspaceBench's four regex multi-token families at L42 (n=400), the target information in NLA and Oracle Lens readouts is recoverable from the prompt text alone: a text-only LLM given the prompt names the target more often than the same LLM given only the readout (0.89 vs 0.60/0.56, same template, k=1), and adding the readout to the prompt does not help (0.88 vs 0.89); the residual readout-only hits (4 NLA, 2 OL of 400) are all regex-form artefacts."
- **Avoid** "the readers re-solve from text" (the data can't show the mechanism) and "readouts contain nothing beyond text" (only these families' targets were tested). Report the 0.47 vs 0.46 tie with the instruction caveat.
- **Correction:** 02's language direction is **Russian vs Polish** (not French/German as written in D10 and handoff #4). The FR/DE version is only the 03 toy.
- **Deck built:** `deck/mats_nla_workspace.pptx` (+ pdf), 60 slides (53 main + 7 appendix), speaker notes on every slide, native charts, Google-Slides-safe fonts, rebuilt by `node deck/build.js` from `deck/content/*.js`. Part D was flagged as pending check (e), which is now resolved, so update it.

## 2026-10-06 — D23: natural col vs TextNLA (check resolved)
On the same 160 col selection items with the frozen templates (unchanged, sha256 0915…923b):

| Reader | Reader @42 | Text-only Qwen | Gemini (text-only) | Reader after regressing out Gemini |
|---|---|---|---|---|
| NLA | 0.60 | 0.48 | **0.97** | 0.51 |
| OL | 0.55 | 0.62 | **0.97** | 0.47 |

- Long vs short lines are visible in the text without counting. Teacher-forced base Qwen is a weak text ceiling for col, so Gemini is the relevant one. With Gemini's signal regressed out, the NLA's small natural signal disappears; OL is at chance throughout.
- **Natural col AUROC says nothing about activation reading.** The col claim comes only from steering, and any natural number is reported next to the 0.97 text ceiling.
- Gaps: Gemini covered 7/14 col templates; col_break is Qwen-only (all at chance). Spend under $5 (shared key, so not separable exactly).

## 2026-10-06 — D24: round-5 pilot cells (excluded split): the "neither decoder reads col" wording is RETRACTED
- **Behavioural validation passes.** At L42 q90 the newline log-odds shift is +1.44 raw (z +86) and +0.72 for 4r* (z +55); generation wrap shifts −6.8 chars raw and −1.55 for 4r*. The 4r* edit changes behaviour yet is invisible to J-Lens (EOL 0.50; content sets clean, |z| ≤ 2.1).
- **NLA with the mined template family f07_count tracks col:**
  - raw: 0.71 (z +5.0/+4.5);
  - **4r* r11: 0.67 (z +3.7/+3.3)**;
  - 4r* q90: 0.63 (conservative z +3.0/+2.6, borderline).
  - It rests mostly on ONE template ("near the end of a hard-wrapped line", 0.66–0.69). Fixed pilot templates and f07_break are flat.
- **OL's newline-continuation (nl_cont) tracks col:** raw 0.76 (z +6.6); 4r* 0.67–0.72 (conservative z +3.9). The shift is linear in dose, and J-removal keeps ~78% per unit dose, close to the model's own behavioural ~70%.
  - **Adapter OFF** (base Qwen in OL's carrier prompt): 0.40–0.44, slight anti-tracking. The effect needs the LoRA.
  - Describe it as **"OL's continuation channel carries the newline decision"**, NOT "OL reports line position": every descriptive OL col template is flat or anti-tracking. A LoRA re-instating the activation for Qwen's own next-token circuitry is not ruled out.
- **Tense 4r* with mined templates:** OL 0.66–0.79 (z +12 to +17) ≥ NLA 0.59–0.74. **D20's "NLA > OL on tense 4r*" does not survive**, so re-specify D21 item 6. OL's tense candidates are continuations too; the adapter-off check for tense hasn't been run.
- **The digit cell is unusable** as an in-J number control: G2 fails (enrichment ≤ shuffled labels) and G4 fails (0 flips). It's readable but causally inert (it probably encodes list length). The count-calibration gap stays open.
- **The NLA "nan" was an analysis bug** (stdout was generated before the NLA jobs finished; `cell()` checked one arm only). Fixed.

**Lead's assessment (self-critique):**
1. **D21's sentence "neither reads col in any phrasing" was premature** and is retracted. It rested on fixed templates. With native/mined phrasings, both decoders show leak-free sensitivity to the J-removed col direction: NLA via one descriptive template, OL via its continuation channel.
2. **Forking paths:** across rounds the pilot tested many templates × directions × doses. The new positives are exactly what selection could produce. They came from pre-specified families (f07 frozen; nl_cont fixed at 00:53, before scoring), but the main run must name **one primary template per reader** in advance, correct for multiplicity, and replicate on fresh items.
3. **The emerging, more honest picture:** sensitivity to non-J content exists but is **channel- and phrasing-specific**. The decoders' *descriptive* language mostly doesn't express it; the NLA has one phrasing that does; OL's continuation (next-token) channel does. That is a nuanced, testable claim for the main run, and it bears directly on "can these tools be used to *report* non-workspace content?".

## 2026-10-06 — D25: Fable round 6, all accepted (FINAL pre-freeze design)
- **Pilot summary:** "No robust descriptive reading; two *candidate* narrow channels, pre-registered for confirmation."
  - OL continuation = the real candidate (z +6.6 raw, +3.9 at 4r*, linear in dose, adapter-dependent, pre-fixed pair).
  - NLA = one template, borderline.
- **Dose fork closed:** **q90 is the primary dose** (behaviourally validated); r11 is sensitivity. Choosing after seeing both would contaminate the NLA cell.
- **OL mechanism:** the 78% vs 70% gain match favours *re-instatement* (the LoRA re-instates the activation and Qwen's own circuitry acts on it) over reporting. Tests (pilot split):
  - (a) **Indentation** (non-J by G2, causally inert by G4): steer it in OL's carrier. A continuation shift ⇒ reading beyond what the model acts on; flat ⇒ consistent with acting.
  - (b) **Gain ratio** (OL continuation shift / model behavioural shift) for col, tense and two random directions with nonzero behavioural effect. Roughly constant ⇒ re-instatement.
  - (c) **Tense adapter-off**, for symmetry.
  - Framing: OL *is* the model plus a LoRA, so it reads what the model reads and says what the model can say. Inheriting col's unverbalisability is a result about the method. A newline-probability shift is usable only if you already know to measure it, so it's not an interpretability readout.
- **Confirmatory design** (prereg revision 4):
  - **4 primaries, Holm over 4:** NLA col (the named "hard-wrapped line" template, q90, 4r*, one direction), OL col (nl_cont, q90, 4r*), OL tense 4r*, NLA tense 4r*.
  - Claim criterion: the primary survives Holm at |z| ≥ 3 AND the declared-secondary family mean agrees in sign.
  - Everything else exploratory and labelled.
  - **Pre-registered negatives:** OL descriptive col templates flat (an equivalence test with a SESOI).
  - **Template-free check:** a pre-written judge question over free NLA readouts ("does this mention position within a line or line length?") on fresh items.
  - Effect sizes with CIs on fresh items.
  - **Templates frozen literally:** no more mining, no new families.
  - **Disclosure:** the prereg states the pilot's number of looks and that primaries were chosen from it.
- **Headline (both branches):** keep the title "Do NLAs read what the model can't say?"; the abstract's first sentence carries the answer.
  - **Branch 1** (replicates): "barely, in narrow channels, and not in the language they use to describe what they read". Say "channel-specific sensitivity", never "reads".
  - **Branch 2:** "no, on the one natural non-J feature the model uses".
  - Parts A, D and the latent-generated reconstruction stand on either branch.
- **Cut:** the next-digit cell, further template mining, the natural col arm beyond one line, the r11/q90 ambiguity.

## 2026-10-06 — D26: prereg revision 4 (confirmatory) written
- **Structure:** P1–P4 (Holm over 4) → positive controls PC1/PC2 → negatives N1–N5 (TOST, SESOI AUROC 0.55 = fraction 0.10) → template-free judge check T1 → secondaries S1–S9 → exploratory.
- **K = 100 null directions per band for the primaries (approved).** With K=20 the band SD has ~16% sampling error, which is ±0.5 on z≈3.5.
- **Disclosure:** ~514 centred template-level cells in the pilot (~2,000 z-values inspected). P1 was the best of 11 NLA col templates.
- **Power:** the band, not n, limits it. P1's family mean is only z +2.5–2.9, so **winner's-curse shrinkage may fail P1's sign-agreement criterion**; the prereg predicts this honestly. P2 is about +3.9–4.0 at every n.
- **n:** 240 fresh col passages (120 also used for T1), 160 tense, 120 language, all disjoint from the pilot and 07 items.
- **Lead's rulings:**
  - frozen analysis script under `experiments/10_main/`, hash recorded at freeze;
  - `notes/deviations.md` created;
  - tense leak set = the union of both lists.
- **Still open:**
  - leak rule (95% band vs |z| ≥ 3): Fable;
  - must experiments/09 finish before the freeze?: Fable (default: fold in if done by Oct 15);
  - who writes the T1 judge sensitivity set: user (default: the lead);
  - WorkspaceBench issue: user.

## 2026-10-06 — D27: Fable round 7 (final freeze review), all accepted → prereg rev 4.1
1. **Leak rule (a), the two-sided 95% J-complement band**, now with **K=100 for the leak band too** (so false voids drop toward the nominal 5%). **A dose ladder is pre-registered:** if q90 voids, the primary is tested at q80, with no further steps.
2. **S6 (the OL mechanism tests) needn't finish before the freeze,** but P2's interpretation is fixed now, conditional on S6:

   | Outcome | Wording |
   |---|---|
   | P2 replicates and S6 = acting | "the continuation channel carries the model's own newline decision" |
   | P2 replicates and S6 = reading | "OL carries non-J col content in its continuation channel" |
   | P2 fails | fixed negative wording |

3. **P1 stays best-of-11.** Pre-register the *expected* shrunken effect (≈ the family-mean z, AUROC ≈ 0.58 rather than 0.63), so "replicates but smaller" is the predicted outcome. **Mixed case** (P1 fails, the family mean passes) = non-replication of the primary, an exploratory note, and no claim.
4. **TOST reachability (blocker).**
   - The cross-item AUROC SE at n=240 is ≈ 0.037 (Hanley–McNeil), so a 90% CI is ±0.06 and a ±0.05 bound can never close.
   - Define the equivalence test on **the statistic the null band is built from** (the paired per-item statistic), at the same n.
   - **Check power by simulation from the band before the freeze.**
   - If underpowered, **raise n for the negatives** (e.g. 480 col passages; the OL descriptive templates reuse the same passages) rather than loosen the bound.
5. **Validity:**
   - the direction w and the q90 dose are **frozen from the pilot/training split**, never re-fit on fresh items;
   - **null bands are computed on the fresh items at the primary's n**;
   - **T1's judge scores *described* content, not formatting** (OL's continuations contain literal newlines), and its 40-item sensitivity set includes **synthetic positives** that do mention line position;
   - **fresh col passages match the pilot's generator** (wrap widths, source distribution).
6. **The headline has FOUR branches** (P1 × P2 pass/fail). P1 failing with P2 passing is the likeliest.
7. **Freeze once the TOST simulation is done and the four-branch wording is written.**

## 2026-10-06 — D28: prereg rev 4.1 done; freeze-ready (design)
- **TOST is defined on the band statistic** (per-item effect vs the K=100 band on the same fresh items), with δ_f frozen from the pilot (1.9–5.1 band-direction SDs). Power to declare all five negatives flat is **1.00 at n=240** under a true zero, and 1.00 under 0.5×SESOI. **n stays 240.**
- **Lead's edits to the agent-drafted wording:**
  - branch 1b now reads "OL only through its next-token continuation channel" (neutral; the S6 clause carries the interpretation);
  - the T1 Gemini max_tokens goes from 8 to 600.
- **Remaining before the freeze timestamp (bookkeeping, Oct 7–18):**
  1. compute the q80 dose (needs one small run);
  2. fix the null-direction seeds;
  3. build the fresh-bank generation + sha256 manifest;
  4. write the 40-item T1 sensitivity set and run the judge gate;
  5. write and hash `experiments/10_main/main_analyze.py`.

## 2026-10-07 — D29: experiment 09 (OL mechanism, pilot split, excluded): mixed; two pre-freeze issues
40/40 jobs finished; candidates unchanged (all 6 sha256s match). Cost ≈ 32.7 A100-h ≈ $82 (a lower bound).

**Results:**
- **(b) Gain ratio** (OL continuation shift / model behavioural shift, r108, adapter on):
  - col raw 0.99 [0.83, 1.20] and col 4r* 1.06 [0.89, 1.26]: one-for-one with the model;
  - J-removal keeps 0.79 (OL) vs 0.75 (model), replicating round 5;
  - random directions with behaviour: 1.65, 1.89; tense raw 3.40.
  - **Not constant across directions.**
- **Tense 4r*: the model's own next-token tense readout does NOT move** (−0.04 nats, z −1.4/−2.8), yet OL tracks it strongly (z +12 to +16). So its gain ratio is undefined (−15.5).
- **(a) Indentation** (non-J, causally inert): sub-threshold, z max +2.7/+2.6. It leans toward reading (OL's shift grows with dose up to AUROC 0.73 at r216, while the model's indentation readout stays flat) and vanishes with the adapter off.
- **(c) Adapter off:** tense collapses like col (f07_tense 0.49–0.51, from 0.65–1.00). Both channels need the LoRA.
- **Sanity:** OL nl_cont on 80 fresh items is 7–12% smaller than round 5 (within noise); the model's newline effect reproduces.

**Verdict:**
- **The literal prereg rule says READING, but only through the undefined tense-4r* ratio, which is an artefact.**
- With a valid-denominator fix (directions kept only if the model's behavioural shift is ≥ +3 z, as already required for the random directions), the verdict is **INCONCLUSIVE**: the ratios are [0.99, 1.06, 3.40, 1.65, 1.89], so "acting" fails the ±25% band and "reading" fails because col isn't the highest.
- The col-specific evidence (one-for-one gain; retention matches the model's) **leans towards acting**.

**Pre-freeze issues (for Fable):**
1. **The §10.1(b) rule needs a valid-denominator clause and a fixed gain-ratio dose,** and the strict ±25% "acting" band looks unreachable (random directions sit at 1.7–1.9× vs col 1.0×). The amendment is pilot-informed, so disclose it.
2. **P3 (OL tense 4r*) may track a behaviourally inactive direction.** The model's next-token tense readout is flat under tense 4r*. If that holds on fresh items, prereg §4.4.5 forces the wording "reader sensitivity to a behaviourally inactive direction". But a next-token readout may be the wrong behavioural measure for tense (tense shows at later verbs, not necessarily the next token), so the behavioural measure for tense should be fixed before the freeze. This also bears on "reading beyond acting": for tense 4r* (and sub-threshold indentation), OL moves where the model's measured behaviour doesn't.

## 2026-10-07 — D30: Fable round 8: demote the S6 gain-ratio verdict; fix P3's behavioural measure (prereg rev 4.2)
- **Why the gain ratio isn't diagnostic.** The model's shift was measured in the original context and OL's in the carrier context. Under pure re-instatement the ratio is (carrier sensitivity)/(original sensitivity), which varies by direction. A constant ratio was never the prediction, so the ±25% band fails in principle. Random directions at 1.7–1.9 mostly show the carrier is a more perturbation-sensitive context. "OL moves where the model doesn't" (indentation, tense 4r*) is compatible with acting plus context: indentation is overridden by the previous line's whitespace in natural text, and the carrier has no previous line; tense may surface at the next verb, not the next token. col's one-for-one ratio and matched retention are *suggestive* of re-instatement, and no more.
- **§10.1(b):**
  - add the valid-denominator clause (direction kept only if the model's behavioural shift is ≥ +3 z) and fix the dose (0.108 relative norm);
  - **no new acting criterion** (it would read as a rule tuned to return "not acting");
  - replace the binary verdict with **three pre-specified descriptive quantities**: (i) col's gain ratio with CI; (ii) J-removal retention, OL vs model; (iii) OL's response on directions the model doesn't act on, with "inert" defined by the generation measure below.
  - Fixed wording: "consistent with re-instatement for col; the channel is not a pure re-emission of the model's measured next-token decision in general."
  - **Disclose in §1.4** that the acting/reading criterion was dropped after the pilot because the test was found to be confounded by context.
- **P3's behavioural measure:** the tense of the **next verb in a greedy continuation, up to 32 tokens**, tagged by a frozen tagger, in the original context. The reason for the change (the pilot's next-token result) is recorded. Conditional wording:
  - if the model acts on tense 4r* by this measure: "OL tracks distributed tense content the model uses";
  - if it doesn't: "reader sensitivity to a direction the model doesn't act on, by two measures" (not softened).
- **Branch 1b wording:** "OL carries col only in its continuation channel, one-for-one with the model's newline decision; the same channel also tracks tense content the model's measured behaviour does not act on, so it is not merely the model's decision re-emitted."
- **Operative fact for Neel, whichever way the mechanism falls:** a next-token probability shift in OL's continuation isn't a readout anyone would find without already knowing what to measure.
- **Post-sprint ideas (not in scope):** a carrier-matched behavioural measure (steer at L42 inside the carrier with the adapter off), or ablating the model's newline-decision components inside OL's forward pass.
- **Freeze** once the three S6 quantities and the P3 measure are written in.

## 2026-10-07 — D31: prereg rev 4.2 done; design COMPLETE
- Revision 4.2 applies D29–D30 (S6 descriptive; P3/P4 next-verb behaviour measure; 1b wording; disclosure paragraph).
- **Lead's edits:**
  - the Branch 1b sentence is made **conditional** on the §4.4.5 next-verb measure (it no longer presumes the model doesn't act on tense);
  - the tense tagger is pinned to **spaCy 3.8.7 + en_core_web_lg 3.8.0** (already used in exp 02) instead of the new `sm` model.
- **Remaining before the freeze timestamp (bookkeeping only):** q80 dose; null seeds; fresh-bank sha256 manifest; the T1 sensitivity set plus judge gate; the tagger wheel sha256; the `main_analyze.py` hash.
