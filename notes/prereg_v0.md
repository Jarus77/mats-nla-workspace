# Pre-registration v0: Do NLAs read non-workspace content, or re-solve from the text?

Draft updated 2026-10-06, **revision 3**. Revision 1 applied the lead/Fable design revision, revision 2 applied Fable's round-2 critique, and revision 3 applies the results logged in `decisions.md` D8–D12 (§0, rows 23–35). All three revisions were approved by the user. Model: Qwen3.6-27B. Status: **draft for the user and Fable**. Freeze target: v1 on Oct 18, before any main-run data. Open choices are marked **[DECIDE: …]**. Companion notes: `pilot_deep_dive.md`, `lit_review.md`, `redteam.md`, `phase1_summary.md`, `wsbench_internals.md`.

---

## 0. Changes from the v0 design (and why)

| # | v0 design | Revised | Rationale |
|---|---|---|---|
| 1 | Per-claim 4-bucket partition: prompt / workspace / non-workspace / unsupported | A **2×2: content location** (J-space / non-J) **× reader mechanism** (tracks activation / tracks text), plus "no content / unsupported". Attribution is **population-level, by intervention sensitivity.** | Per-claim causal attribution is not identifiable: one claim cannot be intervened on in isolation, and judges cannot tell "re-solved" from "read". Intervention sensitivity measured over a population is identifiable. |
| 2 | Tool (B), split-injection, as a core arm | **Cut.** Optional 20-item replication only if time remains. | Viswanath already has the 70B result. It is the concurrent pilot's stated next step. Injected vectors are the most OOD condition, and at-site injection tests only decoding capability (`pilot_deep_dive.md` Part B). |
| 3 | Three co-equal tools | **Headline = (A).** Dose-response steering along *natural* probe directions for 2–3 natural non-J features, calibrated against 2 in-J features. Each reader is reported as a **fraction of probe-recoverable information**. | Natural activations, ground truth that does not depend on any reader, text held fixed. This is the cell nobody has run (lit review §2). |
| 4 | Non-J = probe-decodable **and** not self-reportable (gate) | **Non-J = the probe survives J-space removal, plus the feature is absent from the J-Lens.** Self-report is **measured and reported, not used as a gate.** | Phase 1 showed that self-report failure is often elicitation failure (v1/v2 prompts, the colour/noun format, countries). Gating on it would select items by prompt quality (redteam O2). |
| 5 | (C) planted intermediates via J-Lens directions or full-residual patches | (C) is the **re-solving control**. Plant intermediates by **moving the feature along its probe direction**. Never add J-Lens token directions; never patch full residuals from counterfactual prompts. | Adding J-Lens directions favours the J-Lens by construction (the pilot's swap). Full-residual patches also move the decodable prompt content, so a re-solver would "follow" them. |
| 6 | Doses as a fraction of median residual norm | Doses at about the **90th percentile, the 99th percentile and 2× the 99th percentile** of the natural probe-projection spread. Report the steering norm as a fraction of ‖h‖ and the next-token KL. **OOD check via the NLA reconstructor's FVE**, with a pre-registered margin. A dose where FVE collapses is void. | This keeps steering on-distribution and makes the OOD threshold falsifiable. |
| 7 | Forced-choice candidate log-probs; "judged precision" | **Paired log-odds** between two candidates differing only in the feature value, each written in the reader's native format. **AUROC against a matched null** as the common currency. Judged precision is renamed **"text-verifiable precision"** (unsupported by the text ≠ false). | Removes format asymmetry and reader priors. Avoids calling unverifiable claims hallucinations. |
| 8 | TextNLA on the prompt only | Plus a **re-solving ceiling**: the NLA's *own explanation* goes to a text-only LLM, which is asked for the feature value. | Measures how much of the NLA's apparent success its reconstructed prompt content can explain. |
| 9 | OL at 40/44 vs NLA at 42 | The same, plus **probe accuracy for every feature at 40/42/43/44**, including the `hidden_states[42]` vs block-42-output convention. | Separates a layer confound from a reader effect. Covers the NLA off-by-one risk. |
| 10 | Stacked-bar headline | Headline = **per-reader AUROC on natural non-J features vs the probe ceiling, across steering dose, with error bars.** Panel 2: **what re-solving explains.** The pilot is cited in paragraph 1. | Matches the revised claim; makes the difference from the pilot explicit. |

**Revision 2 (Fable round-2 critique, user-approved).** "Rev 1" below means the revision-1 design.

| # | Rev 1 | Rev 2 | Rationale |
|---|---|---|---|
| 11 | Steer at L38 along the J-orthogonalised probe direction; the L38 arm is the headline | (a) Steer along the **J-removed** probe direction: the top-r* J-space is projected out of the steering vector at the *steering* layer. (b) **Leak detector:** the J-Lens at 40–44 must not move. FollowFeature(J-Lens) ≤ 0.55, else that dose is void for non-J claims. (c) **Read-layer steering is the HEADLINE reader-input test.** The vector is injected directly at the reader's input layer: 42 for the NLA, 40/44 for OL. L38 becomes the on-distribution / behavioural validation arm. | The L38 → 42 layers can rotate non-J content into the J-space before the read layers, so a "non-J" positive could be a workspace read. Read-layer injection removes that path. The leak detector catches residual leakage in the L38 arm. |
| 12 | "Probe = reference, not ceiling; fraction > 1 allowed" everywhere | In the steering arm the probe's AUROC → 1 by construction, so the fraction is bounded by 1 there. "Reference, not ceiling" applies **only to the natural arm**, and a fraction > 1 there is interpretable only after H4 has ruled out re-solving. | The probe direction *is* the steering direction. |
| 13 | FVE as an OOD gate | **FVE is a one-sided guard.** Low FVE means the reader is not reading on-distribution input, so the dose is void. High FVE proves nothing. **The real "doses are in range" evidence is K3:** in-J calibration tracking at the *same* dose. KL and "probe decodes the shift at the read layer" checks are kept. | Reconstructors can extrapolate. Only a positive control at matched dose shows the readers can respond. |
| 14 | J-Lens as a third competitor everywhere | **The J-Lens is not a competitor in the non-J arm.** It sits at ≈0.5 by construction (G3 plus J-removed steering), and its role there is the **leak detector**. It **is** a competitor in the in-J arm. | Avoids a trivially "losing" reader inflating the contrast. |
| 15 | Clamp one probe direction for FollowText | **Clamp validation:** retrain a probe on the clamped activations; its AUROC must be ≤ 0.55. If not, clamp a small subspace (the top-k probe directions found by iterative nullspace projection, INLP) until it is. | A one-direction clamp can leave the feature decodable from other directions, which would contaminate FollowText. |
| 16 | H2 prior 0.25; H2 the headline | **H2 prior 0.40.** Hong implies the NLA must encode non-J content to reconstruct; the open question is whether the verbalizer conditions on it. A fraction ≥ 0.25 is the hard bar. **Co-headline = NLA vs Oracle Lens under text-fixed steering (H3 + H4).** | A bare H2 positive is roughly a Qwen replication of Viswanath with better ground truth. The NLA-vs-OL contrast is the contribution. If both read non-J, "workspace-bound" describes neither reader. |
| 17 | r* by readout agreement | Keep the readout-agreement rule, but **G2 must pass at r* AND at 4r***; show curves over r. **Wording rule for the write-up: "outside the top-r J-space", never "outside the workspace".** | Robustness to the arbitrary rank; avoids overclaiming. |
| 18 | G4 recommended as a gate | **G4 stays a gate.** Ablation and steering use the **J-removed** direction. Features that pass G1–G3 but fail G4 go to the appendix as "decodable, unused". | Behavioural use must be shown for the same direction we steer. |
| 19 | IJ1 not explicitly gated; IJ2 = multihop bridge category; NJ3 truth = the actual next word | **IJ1 is gated like the others.** **IJ2 = a WorkspaceBench family with a known J-Lens token readout and regex ground truth:** language identity (multilingual_mt) or the typo-correction target. **NJ3 ground truth = the model's predicted POS** (from its next-token distribution). | Matched gating; a validated in-J family; NJ3 ground truth defined by the model's own state rather than by text. |
| 20 | Split-injection cut entirely (optional replication only) | A **secondary Viswanath-style J-only / non-J-only projection arm is included**, capped at **half a day**, **appendix by default**. Caveat: norm-matched projections are themselves OOD. | A cheap comparison point to Viswanath and his follow-up. |
| 21 | NLA iter 300 vs 400 undecided; K0 by cosine | **NLA iter_000400 primary, iter 300 as robustness.** **K0:** cosine is a weak discriminator. **Functional tests are primary:** the FVE match (0.761 vs about 0.76 on the card), WorkspaceBench typo_mt (0.73 vs the stored 0.75) and multilingual_mt (0.42 vs 0.44). Fix the DeltaNet kernels first. If cosine stays at about 0.974, proceed with the FVE-matching convention and run a sensitivity analysis under both conventions on the headline arm only. | Cosine near 0.97 cannot separate adjacent layers. A functional match on stored numbers can. |
| 22 | Figure 1 = AUROC vs dose, all features; Figure 3 = the 2×2 | **Figure 1 = dose-response, FollowFeature vs dose, three readers, one non-J and one in-J feature side by side.** The 2×2 becomes an **intro schematic**. **New ≤1-page section, "Re-analysis of the concurrent pilot" (§15).** The pilot and Viswanath are cited in paragraph 1. | The figure carries the claim; the re-analysis motivates the design from existing data. |

**Revision 3** (D8–D12: pipeline smoke, feature gates, the anti-tracking analysis, and Fable round 4).

| # | Rev 2 | Rev 3 | Rationale |
|---|---|---|---|
| 23 | r* by rule, J artefact undecided | **r\* = 384, 4r\* = 1536** (computed by the readout-agreement rule). **Primary lens: neuronpedia n1000.** The camilablank lens (fit on 25 prompts, cosine 0.76 to neuronpedia) is a **cross-check only**; col is gated under both lenses. | D10. |
| 24 | G2 = probe survives J-removal (+ J-Lens absence) | **Final G2.** (i) **Geometric:** enrichment of the diff-in-means direction ≤ 1.5 at r\* and 4r\*. (ii) **Functional:** the J-Lens readout does not contain the feature value **in the form the lens emits**, compared with a shuffled-label null band. **Labels:** NON-J (both pass) / DIFFUSE-BUT-J-READABLE (enrichment ≤ 1.5 but the lens reads it; reported, excluded) / IN-J (enriched and readable). The J-only probe (log-loss / d′ against a random equal-rank band) is secondary. The old survival test is reported **relative to removing a random subspace of equal rank**. | The survival test is uninformative. Every feature keeps ≥ 98% of probe performance under J-removal, topic and language included, and so does removal of a random equal-rank subspace (D8, D10). The functional gate was generalised after pred_pos failed for a test-design reason (the lens emits tokens, not POS labels; D12.6). |
| 25 | ± steering along the J-removed direction (additive) | **Centred construction primary:** h' = h − (v·h)v ± Δv. Additive steering is reported. Norm-preserving steering is **not used**: it is a no-op for readers that renormalise their input (OL `16000·unit(h)`; the NLA's norm-matched injection). | Unequal dilution between the + and − sides made OL anti-track spuriously (0.45 → 0.49 under centring; D11). |
| 26 | Matched null = equal-norm random direction, AUROC ≈ 0.5 | **Null bands with K ≥ 20 directions from both classes** (isotropic, and the J-complement subspace), using the same centred ± construction. Results are reported as a percentile and a z-score against each band, **never subtracted**. | Subtracting a noisy null inflates variance (D9.2). |
| 27 | Leak detector: one-sided, fixed 0.55 | **Two-sided leak test against the J-complement null band**, using the J-Lens **mean shift and AUROC** (paired-win bands are degenerate for a linear readout). **The leak set includes the feature value tokens and, for col, the newline/EOL tokens.** **No rank makes a direction J-invisible.** The removal rank only constructs the steering vector; **the leak check is what certifies a dose as non-J.** | The J-Lens *anti*-tracked under J-removed language steering (z −4.3 to −5.4) and was missed by the one-sided 0.55 rule (D8, D11). The newline decision is J-type content (D12.7). |
| 28 | Doses q90 / q99 / 2×q99 equally weighted | **q90 is the primary dose for col**; q99 secondary; 2×q99 is void (it fails the FVE guard). Observed ‖Δ‖/‖h‖ at q90/q99: col 0.11/0.16, tense 0.11/0.16, language 0.14/0.23. | D8, D10, D12.7. |
| 29 | Single headline (A) with an NLA-vs-OL co-headline | **Two co-headlines.** **Part A, "What counts as non-J?"**: the survival test is uninformative, the calibrated gate works, and POS/tense are J-enriched on Qwen. This is framed as a property of the test, with three differences from the J-space paper (model, lens, 25 atoms vs rank 384/1536). **Part B, "col, deep"**: n = 1 stated in the first sentence. Qwen cannot count without CoT, so the re-solving route is closed. Causal split after J-removal: col keeps 79% of its effect, tense 19%. **Figure 1 = whichever part is cleaner.** The NLA-vs-OL contrast (H3) stays inside Part B. | D10, D12.1–2, D12.5. |
| 30 | NJ1–3 (col, tense, POS); IJ1 line-count report, IJ2 language/typo | **NJ = col.** **Backups:** capitalisation, coding language, in-quote state, gender/number agreement of an upcoming noun (FR/DE). **Count-family replication:** indentation, words since sentence start. **In-J calibration:** language and tense (for K3), plus a **NEW in-J count control: list index with numerals in context ("Item 7:")**, gated for IN-J. **col_report becomes a mini-finding:** explicit report did not move the count into J (enrichment 0.59/0.86, J-Lens 0.500, free-answer Spearman 0.42). | D10 gates (POS and tense are J-enriched; col_report failed as in-J calibration); D12.3–4. |
| 31 | Templates mined per reader | **col: qualitative candidate strings are primary** ("near the end of a long line" vs "at the start of a line"); numerals are secondary. **Two separate template families:** "long line" (the count) vs "newline coming" (the decision). Templates are mined on a held-out split and selected by natural-item discriminability (not steering response). A fixed label template is kept as a 4th column, and fraction-of-probe is template-averaged with its range. Both candidates are scored in one batch. | D9.6, D12.4, D12.7. Batch composition shifts scores by up to 0.26 nats (D5). |
| 32 | FollowFeature = cross-item AUROC | **Primary: the paired win rate and per-item shift** (+ vs − on the same item), as z against the null bands. The cross-item AUROC is reported as secondary. | Cross-item AUROC is capped by item priors (D8.1). |
| 33 | Pivot P1 if < 1 non-J feature passes | **Pivot only if K3 fails, or if col AND the in-J count control are flat for all readers** (→ P1 with a strong Part A). | D12.8. |
| 34 | – | **A declared pilot split (`experiments/08_col_pilot`, 60 items) is excluded from the main analysis.** | Prevents analysis choices being tuned on test data. |
| 35 | OL at 40/44 primary | **L42 is primary for all three readers; OL at 40/44 is sensitivity** (OL at L42 is no worse than at 40/44 on WorkspaceBench). Layer convention = block-42 output (K0 passed). | D5–D7 (already reflected in §4.4/§5; recorded here for completeness). |

---

## 1. Research question

Neel: *"NLAs seem to capture more than J-lens/Oracle Lens. Is that really the non-workspace stuff?"*

**Operational question.** At matched depth (≈ layer 42 of 64) on natural text:
1. Does each reader's output **follow a feature when we move it in the activation with the text held fixed**?
2. Does it **follow the text when we move the text with the feature clamped**?
3. Does this differ between features that live **outside** the J-space and features that live **inside** it?

Readers: NLA (layer 42 only), Oracle Lens (OL; trained at 20–60 step 4), J-Lens.

**Where we stand relative to prior work** (paragraph 1 of the write-up cites both):
- The concurrent MATS pilot `LittleMeHere/nla-vs-lenses` found that on WorkspaceBench intermediates OL ≥ NLA ≫ J-Lens, and read the residual as prompt re-solving.
- Viswanath (Llama-3.3-70B) showed that the NLA names *injected* non-J components that the model cannot report.
- **What's new here:**
  - natural non-J features;
  - causal steering with the text fixed, at the reader's input layer;
  - on-distribution doses;
  - the NLA-vs-Oracle-Lens contrast.

**Two co-headlines (D12).** Figure 1 goes to whichever is cleaner.
- **Part A, "What counts as non-J?"**
  - The standard "probe survives J-removal" test is uninformative on Qwen3.6-27B. All 7 gated features, topic included (0.90 → 0.88), keep ≥ 98% of probe performance, and removing a random equal-rank subspace does the same.
  - A calibrated gate (geometric enrichment plus a functional J-Lens test against null bands) does separate features. Under it, POS and tense are J-enriched on Qwen.
  - **Framing:** this is a property of the *test*, not of the J-space paper. Three differences from the paper: the model; the lens (theirs vs neuronpedia n1000; the 25-prompt camilablank fit has cosine 0.76); 25 atoms vs rank 384/1536.
  - The claim is "the test as run cannot support the conclusion", never "the conclusion is wrong". We offer the random-subspace control as a drop-in.
- **Part B, "col, deep."** The first sentence states **n = 1 feature**; generalisation is treated as open.
  - col (characters since the last newline) passes every gate: enrichment 0.67/1.10, J-Lens AUROC 0.500 inside its band, J-removed steering 49× random, not self-reportable.
  - **Qwen cannot count without CoT** (7% exact), so a Qwen-based decoder cannot re-solve col by reconstructing the text and counting. Tracking at leak-free doses is therefore activation reading.
  - Causal split after J-removal: **col keeps 79% of its causal effect; tense keeps 19%.**
  - The NLA-vs-OL contrast (H3) is tested here.
  - **Framing for col:** the count is non-J; the newline *decision* it feeds is J; a reader could read either. Separate template families and a leak set that includes newline tokens address this.

**Wording rule.** Claims say "outside the top-r J-space", never "outside the workspace".

## 2. The attribution frame (2×2 plus unsupported); used as the intro schematic

| | **Reader tracks the activation**: its output follows a feature moved with the text fixed | **Reader tracks the text**: its output follows a text change with the feature clamped |
|---|---|---|
| **Content in the J-space** (in-J features) | "Workspace reading". Expected for all three readers; this is the calibration. | "Re-solving / re-description of workspace-type content" |
| **Content outside the J-space** (non-J features) | **"Non-workspace reading". Neel's hypothesis for the NLA.** | "Re-solving / re-description of non-J-type content". E.g., counting characters in decoded text. |

Plus **no content / unsupported**: output that follows neither manipulation. Measured as the AUROC on the null contrast, plus the text-verifiable precision of free-form claims.

**Indices, per reader R and feature f (population level)**
- **FollowFeature(R, f, dose)**, **primary form:** the **paired win rate** (fraction of items where LO(+Δ) > LO(−Δ)) and the **mean per-item shift** LO(+Δ) − LO(−Δ). Steering uses the centred construction (§4.4). Both are reported as a percentile and a z-score against null bands of K ≥ 20 random directions (isotropic, and the J-complement), never subtracted.
  - **Secondary:** the cross-item AUROC (capped by item priors; D8).
- **FollowText(R, f)** = AUROC with which R separates two texts whose text-implied feature values differ, after **clamping** the probe projection of both to the same value.
- **Fraction of probe-recoverable information** = (AUROC_R − 0.5) / (AUROC_probe − 0.5), on the same contrast.
  - **Steering arm:** the probe direction is the steering direction, so AUROC_probe → 1 by construction and the fraction is ≤ 1. It reads as "how much of a guaranteed-present shift the reader conveys".
  - **Natural arm:** the probe is a *reference, not a ceiling*. A nonlinear reader could exceed a linear probe. A fraction > 1 is interpretable **only after H4 has ruled out re-solving**; otherwise it is attributed to text.
- **Role of the J-Lens.** In the non-J arm the J-Lens is the **leak detector**, not a competitor (§4.4). In the in-J arm it is a competitor.

## 3. Hypotheses (directional, with my prior)

| ID | Hypothesis | Directional prediction | Prior |
|---|---|---|---|
| **H1, calibration (K3)** | OL and the NLA track in-J features (language, tense, list index) under centred, text-fixed steering at the doses used for col. | Paired win rate above the 97.5th percentile of both null bands, and z ≥ 3, at q90 on ≥ 1 in-J feature. The J-Lens tracks too (it is a competitor in the in-J arm). | 0.75 |
| **HA1, Part A: the survival test is uninformative** | The "probe survives J-removal" statistic does not separate features labelled in-J from features labelled non-J by the calibrated gate. | For all gated features, retention minus random-equal-rank-removal retention lies within ±0.02, at r\* and at 4r\*. Already observed at gate stage (D10); **confirmatory replication on held-out items**, and under the camilablank lens for col. | 0.85 |
| **HA2, Part A: calibrated gate labels** | On Qwen, POS and tense are J-enriched (not NON-J) under the calibrated gate, and col is NON-J. Holds under both lenses. | Enrichment ≥ 2 at r\* for POS/tense; ≤ 1.5 at r\* and 4r\* for col; functional J-Lens test inside the null band for col only. | 0.75 |
| **H2, Part B: the NLA reads col** | Under centred read-layer steering of col (J-removed direction), the NLA's output follows the count at leak-free doses. | NLA paired win rate z ≥ 3 against both null bands at q90 for the "long line" template family; fraction of probe information ≥ 0.25 (the hard bar); the leak test passes at that dose. | **0.40** |
| **H3, Part B: NLA vs OL on col** | The NLA tracks col more than OL does. | NLA − OL paired shift > 0 with an item-clustered CI excluding 0 at q90; OL inside both null bands. If both read col, "workspace-bound" describes neither. | 0.40 |
| **H3b, count vs decision** | A reader that tracks col does so in the "long line" (count) family, not only in the "newline coming" (decision) family. | Count-family z ≥ 3. A reader that tracks only the decision family is reported as reading the J-type decision. | 0.5 (conditional on any col tracking) |
| **H4, natural success is text** | On natural (unsteered) col items, each prose reader's discrimination of the count is no better than a text-only decoder's. | Natural AUROC ≤ TextNLA(prompt) + 0.03, and (NLA) ≤ TextNLA(NLA's explanation) + 0.03. Because Qwen cannot count without CoT, both TextNLA arms are expected to be near chance on exact count; the qualitative long/short contrast is the test. | 0.5 |
| **H5, re-solving control on intermediates (C)** | When an intermediate is planted by moving its probe direction (prompt fixed, model answer flips), the prose readers follow the **text-implied** intermediate more often than the planted one. | P(planted) / (P(planted) + P(text-implied)) < 0.5 for OL and the NLA. | 0.6 |
| **H6, dose monotonicity** | Where a reader tracks a feature, tracking rises from q90 to q99 (valid doses only). | Spearman ρ(dose, shift) > 0. | 0.7 (conditional on any tracking) |

Backup non-J features, if any pass the gates, are tested with the H2/H3 criteria as replications. The two count-family backups (indentation, words since sentence start) form a within-family replication of Part B.

## 4. Operational definitions

### 4.1 J-space
- **Primary:** S_r = span of the top-r right singular vectors of J_L (the J-Lens Jacobian at the read layer L; SVD as in Viswanath).
  - **r\* = 384, 4r\* = 1536 (fixed, D10).** r\* is the smallest r at which a J-Lens built from S_r keeps ≥ 90% top-10 readout agreement on basic_readout.
  - Do **not** calibrate r* on any reader under test.
  - **G2 must pass at both r* and 4r*.** Curves over r are always shown.
- **Robustness:** Anthropic's per-activation sparse NNLS of J-Lens vectors (k = 25). Removal subtracts the fitted nonnegative combination.
- **Always report enrichment** = (energy in S_r) / (r/d), d = 5120. A random vector scores 1.0 (Viswanath: concept vectors 1.9×).
- **J artefact (fixed):** `neuronpedia/jacobian-lens` n1000 wikitext is primary. `camilablank/workspace-lenses` (a 25-prompt fit, cosine 0.76 to neuronpedia) is a **cross-check only**; col's gates and HA1/HA2 are rerun under it.

### 4.2 Feature criteria (gates)
Every gate is evaluated at L42 (primary) and at 40/44, with r\* = 384 and 4r\* = 1536. **Every feature is gated, in-J calibration features included.**

| Gate | Test | NON-J requires | IN-J requires |
|---|---|---|---|
| G1 decodable | Held-out probe accuracy (binary) or R²/Spearman (ordinal) | acc ≥ 0.80 / R² ≥ 0.5 **[as run at gate stage]** | same |
| **G2a geometric** | Enrichment of the **diff-in-means** direction = (energy in S_r)/(r/d) at r\* and 4r\*. INLP top-k is a supplement. | ≤ 1.5 at both | ≥ 2 at r\* |
| **G2b functional** | **The J-Lens readout contains the feature's value in the form the lens emits** (tokens; e.g. classify top-10 tokens by POS for POS, number/length tokens for counts, language-specific tokens for language), on natural items, vs a **shuffled-label null band** | Inside the band | Above the band |
| G2 secondary | J-only probe (trained on P_S h) by log-loss / d′, vs K = 10 random equal-rank subspaces | Inside the random band (reported) | Above (reported) |
| G2 reported only | Old survival test: probe retention under J-removal **minus retention under random equal-rank removal** | – | – |
| G4 causal use (gate) | At q90/q99, steering/ablating along the **J-removed** direction (raw w for in-J) shifts the target behaviour beyond the J-complement random band (item-paired, p < 0.01) | Pass | Pass |
| Reported, not gated | Self-report: free answer (Spearman, MAE vs predicting the mean), answer-slot logit 2AFC | – | – |

**Labels**
- **NON-J** = G2a and G2b pass, and G4 passes.
- **DIFFUSE-BUT-J-READABLE** = G2a passes (not enriched) but G2b shows the lens reads it. Reported, and excluded from non-J claims. Example: bracket depth, borderline.
- **IN-J** = enriched and readable.
- Features that pass G1–G2 but fail G4 go to an appendix as **"decodable, unused"**.

**Gate-stage status (D10, before the final functional re-run for POS):**

| Feature | Label | Key numbers |
|---|---|---|
| col | **NON-J** | R² 0.655; enrichment 0.67/1.10; J-Lens 0.500 [0.499, 0.501]; J-removed steering 49× random, keeps 79% of the full effect |
| col_report | non-J; failed as in-J calibration → **mini-finding** | R² 0.88; enrichment 0.59/0.86; J-Lens 0.500; free-answer Spearman 0.42; 7% exact count |
| bracket depth | diffuse-but-J-readable (borderline) | enrichment 0.57/1.16; J-Lens 0.5035 |
| predicted POS | J-enriched; functional gate to be **re-run with POS-classified lens tokens** (D12.6) | enrichment 2.62/1.56 |
| tense | IN-J (keeps 19% of its causal effect after J-removal) | enrichment 5.6/2.3; J-Lens 0.52 |
| language | IN-J | enrichment 3.3/1.7; J-Lens 0.62 |
| topic (reference) | IN-J | enrichment 6.0/2.3; J-Lens 0.83 |

**Steering direction:** the **J-removed** diff-in-means/probe direction u = normalise((I − P_{S_r*})w) for non-J features (4r\* robustness); raw w for in-J features. This only *constructs* the vector. Certification as non-J at a given dose comes from the leak test (§4.4).

### 4.3 Feature set

| Role | Feature | Read position | Status |
|---|---|---|---|
| **NJ (primary)** | **col**: characters since the last newline during automatic linewrap; ordinal | The token before each newline plus mid-line positions (cluster by passage) | NON-J (D10) |
| NJ backups, non-count | capitalisation state; coding language; in-quote state; **gender/number agreement of an upcoming noun** (FR/DE) | Per feature | To gate (`experiments/04_backup_features`) |
| NJ backups, count family (within-family replication) | indentation; words since sentence start | Per feature | To gate |
| **In-J calibration (K3)** | language identity; tense | As gated | IN-J |
| **In-J count control (NEW)** | **list index with numerals in context** ("Item 7:") | The token after the index | To gate **for IN-J**. This is the count-family positive control that col_report failed to be. |
| Mini-finding | col_report (explicit count question) | Last prompt token | Explicit report did not move the count into J on Qwen (a model difference vs the paper's Claude result) |

### 4.4 Steering
**Two arms**

| Arm | Where the vector is added | Role |
|---|---|---|
| **R (headline): read-layer steering** | At the reader's input layer and position: L42 for all three readers (primary; D7); L40/L44 for OL as sensitivity. Layer convention = block-42 output (K0 passed, D5). Added to the captured vector before it is handed to the reader. No model forward pass after the edit. | **Reader-input test.** Does the reader's output follow a feature moved in its input, with everything else fixed? No downstream layers, so no leak path into the J-space. |
| **U (validation): upstream steering** | L38 at the read position(s), propagated through the model to 40–44. Also at later positions for behaviour (G4). | **On-distribution / behavioural validation.** The shift is produced by the model's own computation. Leak-prone, hence the leak detector. |

**Direction:** J-removed direction u (§4.2); raw w for in-J features.

**Construction: centred, primary.** h' = h − (v·h)v ± Δv, which gives equal norm on both sides and removes the item's own projection.
- Additive steering (h ± Δv) is reported as secondary.
- **Norm-preserving steering is not used.** It is a no-op for OL (`16000·unit(h)`) and the NLA (norm-matched injection at block 1), which renormalise their input.
- The anti-tracking seen under additive ± steering was unequal dilution, not a reader property (D11).

**Doses**
- Let p be the natural projection u·h over the item bank, and m its median. Δ ∈ {q90 − m, q99 − m}, both signs, plus 0.
- **q90 is the primary dose for col**; q99 secondary. **2×q99 is void** (it fails the FVE guard; D8).
- Observed ‖Δ‖/‖h‖ at q90/q99: col 0.11/0.16, tense 0.11/0.16, language 0.14/0.23.

**Null bands (no subtraction)**
- K ≥ 20 random directions from each of two classes, isotropic and the J-complement subspace (orthogonal to S_r\*), with the same norm and the same centred ± construction.
- Every reader statistic is reported as a percentile and a z-score against each band.

**Report per dose**
- ‖Δ‖/‖h‖.
- Next-token KL (arm U; ≤ 0.008 at q99 in smoke).
- Fraction of the dose reaching 40–44 (arm U; 54–86% in smoke).

**Leak test (pre-registered; two-sided; certifies non-J doses)**
- **Statistic:** the J-Lens **mean shift and AUROC** between the ± sides, on the **leak set**: the feature-value tokens (number words/digits and length words for col), **plus the newline/EOL tokens for col**, since the newline decision is J-type content. Computed at layers 40/42/44, for arm R and arm U.
- **Rule:** a dose is **void for non-J claims** if either statistic falls outside the central 95% of the **J-complement null band** in *either* direction (anti-tracking counts as a leak).
- Rationale: no rank makes a direction J-invisible (no spectral elbow), so the leak check, not the removal rank, certifies the dose.
- Precedent: the J-removed language direction fired this test at z −4.3 to −5.4 (D11), so the toy's NLA tracking (+8.3 SD) cannot be claimed as a non-J read.
- **Col's leak test must be redone with centred steering and the calibrated band before any Part B claim.**

**FVE: a one-sided guard**
- NLA reconstructor FVE on steered vs natural activations, same items (OL AR for OL).
- A dose is **void** if mean FVE(steered) < mean FVE(natural) − 0.05, or < the 5th percentile of natural per-item FVE. **[DECIDE: margin.]**
- High FVE proves nothing about being in range. **The evidence that doses are in range is K3:** in-J calibration features tracked at the *same* doses.
- **[VERIFY: an AR checkpoint for the ceselder NLA exists and loads; else use the OL AR as the shared gauge.]**

**FollowText (text moved, feature clamped)**
- Pair texts i and j whose feature values differ (e.g. a 40-char vs a 60-char line).
- Clamp each activation's projection to a common value c (the bank median): h' = h + (c − u·h)u, at the read layer (arm R) or L38 (arm U).
- **Clamp validation (pre-registered):** retrain a probe on clamped activations (held-out split). Its AUROC must be **≤ 0.55**.
  - If it is not, clamp the subspace spanned by the top-k probe directions found by iterative nullspace projection (INLP; k ≤ 8), until a retrained probe is ≤ 0.55.
  - Report k, and check that the clamped subspace's energy is ≤ 5% of ‖h‖². **[DECIDE: k cap.]**

### 4.5 (C) Re-solving control: planted intermediates
- **Items:** WorkspaceBench multihop (100) plus a constructed two-hop bank. Target 300 candidates, ≥120 kept.
  - Chain arithmetic is a secondary option, but no reader is above chance at L42 (pilot), so only if a ≥42 cell exists. **[DECIDE]**
- **Probe:** a multi-class bridge-identity probe at layer L_p ∈ {30, 34, 38}, trained on paraphrase-diverse prompts sharing a bridge.
- **Planting:** move the projection from the own bridge onto a partner bridge along the probe directions (difference of class means), at doses inside the natural spread. Do this in arm R (at the read layer) and arm U (at L_p, with answer flips). **No J-Lens token directions; no residual patches.**
- **Keep an item only if:**
  - (i) the model's answer flips to the partner's answer in ≥50% of planted runs;
  - (ii) the **prompt content stays decodable**: a subject-identity probe at L42 still reads the original subject (AUROC ≥ 0.9), and TextNLA-on-NLA-explanation recovers the original subject at the same rate as unplanted.
- **Score:** paired log-odds of planted vs text-implied bridge, in native format, for each reader. TextNLA(prompt) defines the pure re-solving prediction (always text-implied).
- **Split arm (optional) [DECIDE]:** plant using only the J component vs only the non-J component of the probe difference (cf. the J-space paper: 61% vs 28% flips). This gives the content-location axis for intermediates.

### 4.6 Secondary: Viswanath-style projection arm (appendix by default; capped at half a day)
- For each natural feature item, feed the readers the J-only projection P_S h and the non-J-only projection (I − P_S)h, both norm-matched to ‖h‖, at r* and 4r*.
- Score with the same paired log-odds.
- Compare with Viswanath's follow-up (full 0.55 / J-only 0.18–0.38 / non-J 0.53–0.57 on Llama).
- **Caveat:** norm-matched projections are themselves OOD (a non-J-only vector has never occurred naturally). FVE is reported. This is not used for any headline claim.
- The 20-item concept split-injection replication remains optional (D11).

## 5. Readers, layers, matched comparisons

| Reader | Layers read | Primary cell | Notes |
|---|---|---|---|
| NLA `ceselder/qwen3.6-27b-nla-rl` | 42 only | 42 | **`av_rl_adapters/iter_000400` primary** (pilot, wsbench); **iter_000300 robustness** (model card's guess). Layer convention fixed by the functional K0 (§9). |
| Oracle Lens s3d-rl600 (open model-repo copy) | **42 (primary; same-layer comparison, D7)**; 40, 44 (sensitivity); full 20..60 step-4 layer curve on natural items | **42** | `{layer}` in the prompt; capture with the adapter disabled. Off-grid 42 is validated: it scores no worse than 40/44 on WorkspaceBench (D6). Report OL's per-family best single layer too (layer flexibility is a real OL capability). Never use per-item any-layer selection. |
| J-Lens (neuronpedia n1000) | 40, 42, 44 (and 20..60 step 4 for presence) | 42 | Label-token direction scores. **In-J arm: competitor. Non-J arm: leak detector only** (≈0.5 by construction). |
| Linear probe | 40, 42, 43, 44 | The read layer of the compared reader | Steering arm: ≈1 by construction (bounded fraction). Natural arm: a reference, not a ceiling. |
| TextNLA (base Qwen3.6-27B, WorkspaceBench prompt-only template plus NLA-style instruction) | – | – | Text only. Two inputs: (a) the true prompt prefix; (b) the NLA's own explanation (the re-solving ceiling). A non-Qwen frontier model as a robustness check **[DECIDE: Gemini 3.8 Flash]**. |
| Self-report (reported only) | – | – | Free text; answer-slot logit 2AFC; few-shot. |

- All readers get identical captured vectors (capture once to disk, wsbench convention plus a 42/43 dual capture).
- The DeltaNet kernels (`causal_conv1d`, fla) are pinned once for every run.

## 6. Scoring

### 6.1 Primary: paired log-odds in native format
For a reader R, activation h and the two candidate values a (true or +) and b (alternative or −):

  **LO_R(h) = log p_R(T_R(a) | prompt_R, h) − log p_R(T_R(b) | prompt_R, h)**, teacher-forced.

The two templates differ **only** in the feature value. Templates are mined from each reader's own free generations on a held-out calibration split and frozen before the main run.
- **NLA:** an explanation-register sentence after `<explanation>`. Example: "The current line is already about {forty} characters long, close to the wrap width."
- **OL:** a `- ` bullet in its native form (often continuation-like predictions). Example: "- {walked} slowly to the door…" vs "- {walks} slowly…" for tense.
- **J-Lens:** max cosine score over each value's label-token variants, s(a) − s(b). In the non-J arm this score is the leak detector (§4.4).
- **Probe:** the projection.
- **TextNLA:** the same NLA-register template, conditioned on the text.
- **Template rules (D9.6, D12):**
  - Mined from each reader's own generations on a held-out split, and frozen before any steering.
  - Selected by natural-item discriminability, not steering response. Same procedure for every reader; OL templates are content continuations, not label bullets.
  - A fixed label template is kept as a 4th column. Fraction-of-probe is template-averaged and reported with its range.
  - **For col:** qualitative candidates are primary ("near the end of a long line" vs "at the start of a line"); numerals are secondary.
  - **Two separate families for col:** "long line" (count) and "newline coming" (decision).
  - **Both candidates of a pair are always scored in the same batch** (batch composition shifts scores by up to 0.26 nats; D5).

**Reported statistics**
- **FollowFeature (primary):** the paired win rate and mean per-item shift of LO between the centred +Δ and −Δ versions of the same item. Each is reported as a percentile and z against the K ≥ 20 isotropic and J-complement null bands (never subtracted). Secondary: the cross-item AUROC.
- FollowText: the same paired statistics between the clamped texts.
- Natural: AUROC between items with high vs low natural values.
- Item-permuted candidates (score against other items' values) as an additional null.

**Statistics**
- Item-clustered bootstrap with 2,000 resamples, clustering by passage, plus a per-feature mixed model `LO ~ dose × reader + (1 | item)`.
- Holm correction across the 3 non-J features within each hypothesis.
- **Primary arm for H2/H3 = arm R** (read-layer). Arm U results are reported alongside, restricted to non-leaked, FVE-valid doses.

### 6.2 Secondary: free-form readouts and text-verifiable precision
- k = 5 samples per cell (pilot single-sample retest κ was 0.55–0.70) at 0-dose and the 99th dose, on 60 items per feature.
- Judges: Gemini 3.8 Flash plus one Fireworks non-Qwen model **[DECIDE: model]**.
  - They (a) extract feature-value claims, giving a judged FollowFeature that cross-checks §6.1;
  - (b) split readouts into atomic claims and label each **text-verifiable-true / text-contradicted / not text-verifiable**.
  - **Text-verifiable precision** = true / (true + contradicted). Unverifiable ≠ false.
- **Judge validation:** the user hand-labels **300 claims**, stratified by reader and label (~3 h). Require Cohen's κ ≥ 0.6 per judge on the 3-way label, and judge-judge κ reported. Below 0.6: revise the prompt once, then fall back to reporting only §6.1.

### 6.3 Re-solving decomposition (panel 2)
For each non-J feature, on natural items:
- AUROC(NLA)
- AUROC(TextNLA | prompt)
- AUROC(TextNLA | NLA explanation)
- AUROC(OL), AUROC(J-Lens)

Plus the **prompt-recovery score** of each readout (judge: fraction of prompt content words recoverable), as a covariate predicting natural success. This was the pilot's untested explanation.

## 7. Controls (summary)
- 0-dose; equal-norm random direction in the same subspace class; isotropic random.
- Item-permuted candidate null.
- Clamp-only (no text change), to check the clamp itself is inert. **Clamp validation:** a probe retrained on clamped activations has AUROC ≤ 0.55, else INLP subspace clamp.
- **Leak test:** two-sided, J-Lens mean shift and AUROC vs the J-complement null band, leak set = value tokens plus newline/EOL for col (§4.4).
- K ≥ 20 null bands (isotropic and J-complement), centred ± construction.
- In-J count control (list index) for the count family.
- Probe-decodes-shift check (arm U).
- In-J calibration features.
- Explicit-report vs automatic condition (col_report vs col; a mini-finding, since explicit report did not move the count into J). The count-family positive control is the list index.
- FVE as a one-sided guard; in-range evidence = K3 at matched doses.
- TextNLA (prompt) and TextNLA (NLA explanation).
- Layer-matched probe ceilings at 40/42/43/44.
- Pilot replication on multihop100 as a pipeline check (expect J 39 / OL 80 / NLA 75, ±4).

## 8. Sample sizes and power
- **Declared pilot split:** `experiments/08_col_pilot` (60 items) is used for development only and is **excluded from every main analysis**. Main-run col items are disjoint at the passage level.
- **Unit:** item (passage or stem). FollowFeature is a within-item ± contrast, analysed as AUROC over items.
- **Hanley–McNeil, two-sided α = 0.05, independent items:**

  | True AUROC | Items per class | Power vs 0.5 |
  |---|---|---|
  | 0.60 | 100 | 0.69 |
  | 0.60 | **200** | **0.94** |
  | 0.55 | 200 | 0.41 |
  | 0.55 | 400 | 0.69 |
  | 0.65 | 100 | 0.96 |

  SE at n = 200, AUROC 0.6: ±0.028.
- **Clustering:** col has up to 4 positions per passage. With ICC ≈ 0.5 the design effect is 1 + (4 − 1)·0.5 = 2.5. So 240 passages give an effective n ≈ 380 positions ≈ 190 per class after splitting.
- **Plan: 240 items per feature**, which detects AUROC 0.60 with power ≥ 0.9. We will *not* claim effects of AUROC ≤ 0.55; that is the pre-declared "no meaningful tracking" region.
- **NLA vs OL difference** (paired, same items): about 0.05 AUROC detectable at n ≈ 240 if the between-reader correlation ≥ 0.5 **[compute exactly on prep data]**.
- **(C):** 120 kept items. P(planted) CI half-width ≈ ±0.09 at p = 0.5. McNemar planted vs text-implied with discordant rates 0.20 / 0.05 needs n = 85.
- **Compute:** about 5 features × 240 items × 7 doses × (3 readers + TextNLA) × 2 candidates × 3 templates ≈ 200k short teacher-forced passes, plus about 15k generations (k = 5) and the AR FVE passes. A few H100-days on Modal; cheap.

## 9. Kill criteria and pivots

| When | Check | Kill / pivot |
|---|---|---|
| **K0** prep (Oct 7) | **Fix the DeltaNet kernels first** (pin `causal_conv1d` + fla). Then **functional tests (primary):** NLA FVE under each convention (match 0.761 vs the card's ~0.76); WorkspaceBench typo_mt (0.73 vs stored 0.75) and multilingual_mt (0.42 vs 0.44) at L42. Cosine vs `example_activations.parquet` is secondary (weak discriminator). Pipeline reproduces the pilot multihop100 numbers ±4. | Functional tests pick a convention → use it. **If cosine stays ≈ 0.974** with no clean separation, proceed with the FVE-matching convention and run a **sensitivity analysis under both conventions on the headline arm only**. Functional tests fail under both → stop NLA arms. |
| **K1** prep (Oct 12) | ≥ 2 non-J features pass G1–G4 under the final G2, and ≥ 1 in-J feature passes. **Status (D10): partially passed.** 1 non-J (col), in-J language and tense; backups being gated. | Run with col plus any passing backup (prereg rule "1 + backup"). Part A stands regardless. |
| **K2** sprint D2 | (a) **FVE guard** (one-sided): AR FVE within margin at q90/q99 (per example, k ≥ 4; detectable shift ~0.04). (b) **Leak test** (two-sided, vs the J-complement band; leak set includes newline/EOL for col). (c) Dose propagation (arm U). | FVE fails at q99 → q90 only; fails at q90 → feature void. Leak at a dose → that dose is void for non-J claims. col void at every dose → Part B becomes descriptive; Part A carries the write-up. |
| **K3** sprint D3 | **Dose-in-range evidence.** Calibration (H1): OL or the NLA tracks ≥ 1 in-J feature (language, tense, or the list-index count control) **at q90, the same dose used for col**. Tracking = paired win rate above the 97.5th percentile of both null bands, and z ≥ 3. Null bands are centred on chance. | Fails → the readout is broken or the doses are too small. Switch to generation plus judged scoring (§6.2). Still fails → **pivot P2**. |
| **K4** sprint D7 | (C) yield: ≥60 items meet keep criteria (i) and (ii). | <60 → report (C) as descriptive, n stated; chain arithmetic at a ≥42 cell if one exists. |

- **Pivot rule (D12.8).** Pivot **only if K3 fails**, or if **col AND the in-J count control are both flat for all readers**. A flat col with a tracked list index is a result (readers can read counts in J but not col), not a pivot.
- **P1:** a strong Part A ("the standard non-J test is uninformative; the calibrated gate; POS/tense J-enriched on Qwen"), plus the col negative and (C), plus the natural re-solving decomposition.
- **P2 (readouts broken, K3 fails):** natural-only analysis (H4) plus (C) plus the claim audit, framed as "how much of each reader's success is explained by text"; Part A unaffected.

## 10. Decisive predictions by outcome

| Outcome pattern (valid doses) | Reading | Answer to Neel |
|---|---|---|
| NLA tracks non-J in arm R (CI > 0.5, fraction ≥ 0.25); OL does not; both track in-J at the same doses; TextNLA cannot explain natural success | The NLA reads content outside the top-r J-space that OL misses | **Yes**, at least for these features (the co-headline) |
| NLA tracks non-J **and** OL tracks non-J similarly | Non-J content is readable by any trained verbalizer; "workspace-bound" describes neither | "Not specific to NLAs" |
| NLA tracks non-J in arm R but not in arm U (leak-free doses) | The reader *can* decode the direction, but natural propagation does not produce a readable shift | Capability yes, natural relevance unclear |
| A positive in arm U only at doses where the leak detector fires | Content rotated into the J-space | Not evidence for non-J reading |
| No reader tracks non-J; all track in-J; NLA natural AUROC ≈ TextNLA | Verbal readers are workspace/text-bound; the NLA's extra content is re-description | **No.** The extra is text, not latent |
| NLA tracks col in the "long line" family at leak-free q90; OL inside the null bands | The NLA reads a count Qwen cannot compute from text and OL does not convey; n = 1 feature | Yes for col; generality open (count-family backups decide) |
| Readers track col only in the "newline coming" family | Readers read the J-type decision, not the non-J count | No for the count |
| col flat for all readers; list index tracked | Readers read counts in J but not the non-J count | **No**, with a clean positive control |
| No reader tracks in-J either | The method cannot see anything | Uninformative → K3 pivot |
| NLA FollowText > FollowFeature for non-J **and** in-J | The NLA is predominantly a text re-solver | No; consistent with the pilot |
| (C): readers follow planted ≥ text-implied | Readers read computed intermediates | Strengthens any H2 positive |
| (C): readers follow text-implied | Re-solving explains intermediate-naming | Consistent with the pilot, now on-distribution |

## 11. Figures

**Intro schematic.** The §2 2×2 (content location × reader mechanism), drawn as a diagram with no data.

**Figure 1 = whichever co-headline is cleaner (D12.1).** Two candidates:
- **1-A (Part A):** per feature, probe retention under J-removal vs under random equal-rank removal (all ≈ 1, the uninformative test), next to the calibrated gate (G2a enrichment at r\*/4r\* and G2b functional z), with labels NON-J / DIFFUSE / IN-J.
- **1-B (Part B):** col dose-response. Paired shift (z vs null bands) vs dose (0, q90, q99) for the NLA, OL and the J-Lens.
  - The J-Lens is drawn as the grey leak trace, with leak-voided doses hollow.
  - Next to it, the in-J count control (list index) and one in-J feature (language or tense).

```
 1-B   col (non-J, "long line" family)          list index (in-J count control)
 z vs                                          z vs
 band  NLA?  ●──────●                          band       ●───── J-Lens
  +3 ┤      ╱                                   +3 ┤   ●──╱─────── OL
     │  ●──╱      OL? ──●                          │  ╱  ●──────── NLA
   0 ┼●─────────●──────○  J-Lens (leak trace)    0 ┼●
     └0────────q90────q99                          └0────────q90────q99
```

**Figure 2 (what re-solving explains).** Grouped bars per non-J feature on natural items: NLA, TextNLA | prompt, TextNLA | NLA explanation, OL, J-Lens, probe. Inset: FollowText vs FollowFeature per reader (the 2×2 mechanism axis).

**Figure 3 (mechanism scatter).** FollowFeature (x) vs FollowText (y) per reader × feature, coloured by content location. This is the data version of the intro schematic.

**Figure 4 (C).** P(planted) vs P(text-implied) per reader, with the TextNLA prediction marked; optional J-only / non-J-only planting.

**Supplementary.**
- Probe AUROC at 40/42/43/44 and its retention under J-removal across r, with enrichment.
- Dose diagnostics: ‖αu‖/‖h‖, next-token KL, AR FVE.
- Self-report rates per feature (reported, not gated).
- Text-verifiable precision per reader.
- The leak detector per dose (J-Lens FollowFeature at 40–44, at r* and 4r*), and clamp validation (retrained-probe AUROC, INLP k).
- §4.6 projection arm (J-only / non-J-only).
- "Decodable, unused" features (passed G1–G3, failed G4).

## 12. Timeline

**Prep, Oct 5–18**

GPU work is allowed in prep. **[DECIDE: confirm MATS rules on pre-sprint work.]**

| Dates | Work |
|---|---|
| Oct 5–6 | **Fix and pin the DeltaNet kernels first.** Modal capture pipeline (capture-once, dual 42/43). Reproduce pilot multihop100. |
| Oct 7 | **K0 (functional):** NLA FVE per convention; typo_mt and multilingual_mt at L42 vs stored numbers; cosine as secondary. Locate the AR checkpoint. |
| Oct 8–9 | Paired log-odds scorer for every reader. Mine and freeze templates on a calibration split. Sanity-check on basic_readout (positive control). |
| Oct 6–12 | **Done (D10):** gates for col, col_report, bracket depth, POS, tense, language, topic at r* = 384/1536. **Remaining:** re-run the POS functional gate with POS-classified lens tokens; gate the backups and the list-index in-J count control; gate col under the camilablank lens; redo col's leak test (centred, calibrated band). Original plan: Train probes at 38/40/42/43/44. J SVD spectrum and the r* rule; G2 at r* and 4r*. Run gates G1–G4 on every feature, including the in-J ones → **K1**. |
| Oct 13–14 | Steering infrastructure for arms R and U. Natural spreads and doses. Leak detector, clamp validation (INLP fallback), FVE guard on 20 items. |
| Oct 15–16 | (C) bank and bridge probes; flip-rate pilot (n = 50). TextNLA prompts. Judge prompts. User labels 300 claims. |
| Oct 17–18 | Dry run of every arm at n = 10. Fable review. **Freeze prereg v1** (commit hash recorded, outside this repo's history if no commits are allowed). |

**Sprint, 2 weeks** **[DECIDE: dates; assumed Oct 19 – Nov 1]**

| Day | Work | Gate |
|---|---|---|
| D1 | Part B: col arm R (centred, q90/q99, both template families), three readers; col leak test; Part A confirmatory gates on held-out items | |
| D2 | Arm U: L38 steering, behaviour, leak detector, probe-shift check, FVE guard; void assignment | **K2** |
| D3 | In-J calibration at matched doses (language, tense, list index; both arms); K ≥ 20 null bands | **K3** |
| D4 | FollowText (clamp, with validation); TextNLA (prompt / NLA explanation); natural AUROCs; §4.6 projection arm (≤ half a day) | |
| D5 | First-pass Figures 1–3; write a one-page interim for Fable/Neel | |
| D6 | (C) planting at full n; behaviour flips; keep criteria | |
| D7 | (C) reader scoring; checkpoint, re-plan | **K4** |
| D8 | Free-form k = 5 generations; judge runs; judge validation κ | |
| D9 | Robustness: r sweep (r*, 4r*), NNLS J definition, OL at 42, NLA iter 300, layer-convention sensitivity (if K0 needed it) | |
| D10 | Robustness: templates, a second TextNLA model, layer 43 convention | |
| D11 | Optional: 20-item split-injection replication (only if D1–D10 are on track) | |
| D12 | Final figures, statistics, error bars | |
| D13 | Write-up: Neel-style summary first; the pilot and Viswanath in paragraph 1; §15 pilot re-analysis section; wording rule "outside the top-r J-space" | |
| D14 | Buffer; red-team the write-up with Fable | |

## 13. Risks
1. **Readers fail for uninteresting reasons.**
   - They were never trained to verbalize character counts or POS, and the paired templates may sit outside their output distribution.
   - A non-J null is then uninterpretable.
   - Mitigations: in-J calibration in the *same* family (list index vs col), templates mined from each reader's own outputs, judged free-form scoring as a cross-check, the K3 gate.
2. **Feature validity.** Natural features may not pass G2/G4 on Qwen3.6-27B at L42, or the probe direction may not be the causal one.
   - Steering may need doses beyond the FVE-valid range to move behaviour.
   - G4 plus the FVE gate may leave few valid cells.
3. **"J-space" is a modelling choice.** A heavy-tailed spectrum with no elbow: on Llama the conclusions flipped between r = 200 and r = 2000, and a random vector puts 2.4% inside r = 200.
   - What counts as "non-J" depends on r and on the J artefact.
   - Mitigations: curves over r, r* = 384 fixed by an a-priori rule, 4r* = 1536 robustness, NNLS robustness, enrichment reporting.
4. **NLA layer off-by-one** (Viswanath's Anthropic NLA uses the `hidden_states[L]` convention). If it is wrong, every NLA number is off-layer → K0.
5. **Novelty and overlap.**
   - Viswanath's natural line count is already negative for the NLA on Llama (3%).
   - His unposted follow-up does reader-side J/non-J reads on natural safety activations.
   - The pilot owns split-injection.
   - Our differentiators: three readers including OL, Qwen, text-fixed steering at on-distribution doses with an FVE gate, the FollowText clamp, TextNLA re-solving ceilings.
   - **[DECIDE: contact the pilot author?]**
5b. **n = 1 non-J feature.** Part B rests on col alone. Mitigations: count-family replication (indentation, words since sentence start), non-count backups, and n = 1 stated in Part B's first sentence.
5c. **Count vs decision.** col feeds the J-type newline decision; a reader could read the decision rather than the count. Mitigations: separate template families, newline/EOL in the leak set, q90 primary dose.
6. **Steering leak.** In arm U, layers 38→42 can rotate non-J content into the J-space. Handled by J-removed directions, the leak detector, and arm R as the headline. Residual risk: leakage into directions outside the top-r* J-space but inside the 4r* tail. The leak detector is also run at 4r*.
7. **Clamp leaves the feature decodable.** Handled by retrained-probe validation and the INLP subspace clamp. Residual risk: a nonlinear encoding a linear probe misses.
8. **OOD despite percentile doses.** Readers normalise input direction (OL unit-norm ×16000; NLA norm-matched add), so even small steers change the direction the reader sees. The FVE gate and KL reporting cover this partly.
9. **Judge validity** for the secondary metrics. Gemini over-credits on wsbench (precision 0.43 vs hand-labelled 0.33).

## 14. Open decisions (collected)
**Resolved in revisions 2–3**
- G4 is a gate.
- NLA iter 400 primary, iter 300 robustness.
- r\* = 384, 4r\* = 1536.
- neuronpedia n1000 primary; camilablank is a cross-check.
- Final G2 (geometric plus functional, with labels).
- Centred steering; K ≥ 20 null bands; two-sided leak test vs the J-complement band.
- q90 primary for col.
- Two co-headlines.
- NJ = col, plus the listed backups.
- In-J calibration = language, tense and list index.
- col templates: qualitative primary, two families.
- Pivot rule (D12.8).
- Pilot split excluded.
- L42 primary for all readers.

**Still open**
- Clamp-validation threshold (0.55) and INLP k cap.
- FVE margin (0.05; per-example, k ≥ 4).
- G1 thresholds for backups (as run at gate stage?).
- Fireworks judge model.
- TextNLA robustness model.
- (C) chain-arithmetic fallback and the J/non-J split planting arm.
- Sprint dates and pre-sprint rules.
- Contacting the pilot author.

## 15. Re-analysis of the concurrent pilot (descriptive; ≤1 page in the write-up)

All numbers are recomputed from the raw JSONL of `LittleMeHere/nla-vs-lenses` (commit `58d34a8`); details are in `pilot_deep_dive.md`. This analysis is descriptive and motivates the design. It is not a test of our hypotheses.

1. **The NLA finds no hidden step that OL misses.** Multihop at L42 (n = 100): the NLA names the bridge where OL misses on **3** items; OL names it where the NLA misses on **8** (McNemar p = 0.23).
2. **The two prose readers succeed on the same items.** phi(OL, NLA) = **0.69**, vs phi(J-Lens, OL) = 0.35 and phi(J-Lens, NLA) = 0.32. That fits both decoding the same prompt-recoverable content.
3. **Bridge hits are solved-fact statements.** The bridge is named together with the answer in **67/75** NLA hits and **79/80** OL hits. The model's own top-1 next token is already the answer on 74/100 items.
4. **Single-sample noise is large.** Readouts on the original vs a near-identity random-direction control flip on 7–10 of 50 items (Cohen's κ = **0.55–0.70**). Differences under about 10/50 are inside this noise, which covers the swap and brew results.
5. **The swap test was weak.** The planting scale escalated until the J-Lens showed the planted bridge, and **34/50 items hit the 16× cap**. The J-Lens shows the planted bridge in only 14–17/50. OL and the NLA follow it in 6/17 and 4/17 of those items.
6. **Survival of deletion is ambiguous.** After deleting about 31 J-Lens directions, 52–59% of the prose readers' bridge mentions survive. Re-solving from the prompt predicts this, and so does reading bridge information outside the top-r J-space. Our FollowFeature (text fixed) vs FollowText (feature clamped) contrast is designed to separate the two.
