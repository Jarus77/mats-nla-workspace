# Pre-registration: Do NLAs read what the model can't say? (confirmatory replication, Qwen3.6-27B)

Draft 2026-10-06, **revision 4: the confirmatory pre-registration to be frozen before the sprint** (Oct 19 – Nov 1). Revisions 1–3 applied D1–D12 (user-approved). Revision 4 applies `decisions.md` D13–D25, with D20–D25 decisive, and restructures the document as a **confirmatory replication of pilot findings**: four primary cells (Holm over 4), two positive controls, pre-registered negatives as equivalence tests, one template-free check, then declared secondaries; everything else is exploratory and labelled as such. Model: Qwen3.6-27B. Status: **draft for the user and Fable; freeze target v1 = Oct 18, before any main-run data.** Remaining choices are marked **[DECIDE: …]** and collected in §14. Revision-3 text is backed up at `scratchpad/prereg_rev3_backup.md`. Companion notes: `pilot_deep_dive.md`, `lit_review.md`, `redteam.md`, `phase1_summary.md`, `wsbench_internals.md`; pilot evidence: `experiments/08_col_pilot/RESULTS.md` (pilot, calibration, round 5), `experiments/07_template_mining/RESULTS.md`.

**How to read this document.** §3 is the binding part: it lists, in order, what is tested, with which statistic, against which threshold. §4–§8 define the operations §3 refers to. Anything in the write-up that is not in §3.1–§3.5 is reported under an "exploratory" label.

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


**Revision 4** (D13–D25: lens robustness, re-solving decomposition, col pilot, planting, template freeze, backup gates, K3, Fable rounds 5–6). Rows 36–60 supersede any earlier row they contradict.

| # | Rev 3 | Rev 4 | Rationale |
|---|---|---|---|
| 36 | Hypotheses H1–H6 in discovery order; H2 "NLA reads col" headline | **Confirmatory structure, in this order:** (1) four PRIMARY cells P1–P4, Holm over 4; (2) positive controls PC1–PC2; (3) pre-registered negatives N1–N5 as equivalence tests (TOST, SESOI); (4) template-free judge check T1; (5) declared secondaries S1–S9; (6) exploratory, labelled. | D21 last item, D25. The pilot tested many cells; only a small, pre-named family can be confirmatory. |
| 37 | Primary test = paired win / shift vs null bands, H2 "CI excludes 0.5" plus z ≥ 3 | **Claim criterion:** a primary is claimed iff (a) z_shift ≥ +3 against **both** null bands (isotropic and J-complement, K = 100 per class at the exact matched norm), and the real shift exceeds all 100 nulls in each band; (b) it survives Holm over the 4 primaries; (c) the mean z of its **declared-secondary family** has the same sign; (d) the cell is not void (leak, FVE). | D15 ("CI excludes 0.5" passes on noise); D21.1 (|z| ≥ 3); D25 (Holm + sign agreement). K raised from 20 to 100 because with K = 20 the band SD has ~16% sampling error, i.e. ±0.5 on z ≈ 3.5. |
| 38 | – | **P1 = NLA col** with the single frozen template `nla:nj1_col:m2` ("…located near the end of / at the beginning of a hard-wrapped line."), centred, regression direction J-removed at 4r\* = 1536, q90, L42. | D24, D25. One template, named in advance. |
| 39 | – | **P2 = OL col** with the newline-continuation pairs `NL_CONT_OL` (r5_templates.py), same cell. | D24, D25. Pair fixed (00:53) before any round-5 scoring. |
| 40 | Tense = in-J calibration only; D21.6 "tense 4r\* NLA > OL" secondary | **P3 = OL tense 4r\*** and **P4 = NLA tense 4r\***, each with its frozen f07_tense family, q90, L42. The NLA-vs-OL tense contrast is **exploratory**. | D24: "NLA > OL on tense 4r\*" did not survive mined templates (OL 0.66–0.79 ≥ NLA 0.59–0.74). **Retracted.** |
| 41 | Leak set: value tokens + newline/EOL for col | **Label set + content set for every feature**, two-sided against the J-complement band at the cell's read layer. Primary sets fire at the 95% band; supplementary sets fire at |z| ≥ 3. A leak voids the dose for non-J claims. | D20 (the label-only set missed the language 4r\* leak, Cyrillic content z +3.7). |
| 42 | q90 primary for col, q99 secondary | **q90 is the primary dose for all four primaries; r11 (0.11‖h‖ per item) is a declared sensitivity, never a substitute.** | D25 closes the q90/r11 fork: choosing after seeing both would contaminate P1 (r11 z +3.7 vs q90 +3.0 in the pilot). |
| 43 | Behavioural validation via G4 at gate stage | **Behavioural validation is reported at each primary cell** (exact direction, dose, layer): col newline log-odds at L42 and the L38 generation wrap shift; tense next-verb tense log-odds. | D21 ("report the realised causal shift where the null/positive is claimed"); D24 (pilot: +0.72 nats, z +55; −1.55 chars). |
| 44 | Matched null = K ≥ 20 per class | **K = 100 per class, fresh seed, at the exact real-cell norm for every primary** (the round-5 plan lacked matched q90 nulls for jrem4/cm directions; the pilot z values are proxies). K = 20 for secondaries. | R5 † bug; band-SD precision (row 37). |
| 45 | H3 (NLA > OL on col), H3b (count vs decision), H4, H5, H6 | **H3 → exploratory.** H3b is absorbed into P1/P2 vs N1–N5 (descriptive templates vs channels). **H4 (natural arm) → one exploratory line**, always next to the Gemini text ceiling (0.97). **H5 → (C) relabelled as in-J calibration**, α ≤ 1.5 primary within (C). H6 → part of S2/S3. | D15 power (ΔAUROC ≤ 0.03 vs bar 0.05); D23 (natural col AUROC is text); D21 + D16 ((C) is in-J). |
| 46 | – | **Pre-registered negatives as equivalence tests:** OL descriptive col families flat at the primary cell; SESOI = AUROC 0.55 (⇔ fraction of probe 0.10 at probe AUROC 1.00), TOST at α = 0.05 per family, intersection-union over families. | D21.2, D25. |
| 47 | Free-form judged scoring (§6.2) for col | **Template-free check T1:** one pre-written yes/no judge question over free NLA and OL readouts on fresh items; Gemini 3.8 Flash primary, kimi-k3 second; expected yes-rate ≈ 0. | D25; base rate 0/4,080 NLA, 0/1,680 OL (pilot); 0/320, 0/960 (07). |
| 48 | Templates "mined and frozen before steering" | **Templates frozen literally:** `templates_frozen.json` sha256 `09156722b059edb99d01e919c29b2ff660e37cba3152f2d3ce67f3272d3e923b`; `r5_templates.py` sha256 `40d18f28…d3ff5`; `templates.py` `4de33584…afb1`; `calib_templates.py` `ce6d758d…488f`. No edits (including the ungrammatical "far into of"), no further mining, no new families. | D25. |
| 49 | – | **Disclosure** of the pilot's number of looks (≈ 514 centred template-level reader cells, ≈ 242 family-level; ≈ 340 / 172 on col alone) and of the fact that P1–P4 were chosen from them (§1.4). | D24.2, D25. |
| 50 | Two co-headlines (Part A, Part B) | **Two-branch headline (D25)** keyed to P1/P2; Part A, Part D and the latent-generated reconstruction (D16) stand on either branch. Title kept: "Do NLAs read what the model can't say?" | D25. |
| 51 | D21 working sentence "neither reads col in any phrasing" | **Retracted.** It rested on fixed templates; with frozen mined / native phrasings both decoders showed leak-free sensitivity in the pilot. | D24.1. |
| 52 | In-J count control = list index ("Item N:"); next-digit cell (D21) | **Both dropped.** List index failed G2/G2b (D19); next-digit failed G2 and G4 (0/30 flips; D24). The count-calibration gap is stated as open; language and tense are the positive controls. | D19, D24, D25 cut list. |
| 53 | NJ backups and count-family replication | **No backup qualifies** (D19). The 16 gate runs (02: 7; 04: 9; ≈ 13 distinct features) become the **positional-bookkeeping survey** of Part A, scoped as a survey, not a theorem. | D19, D21. |
| 54 | Part D (WorkspaceBench) as H4 / panel 2 | **Part D wording fixed (D22):** "target information is recoverable from the prompt text alone" (0.89 prompt-only vs 0.60/0.56 readout-only; prompt + readout 0.88). Not "the readers re-solve". Completed analysis; no new data. | D14, D22. |
| 55 | Planting = re-solving control (C) | **Planting = in-J calibration** (swap direction 6.5× enriched). α ≤ 1.5 subset primary within (C) (109 items at L38). Its contribution is the latent-generated reconstruction (12/12 rewrites), pre-registered as S8b. | D16, D21, D25. |
| 56 | col pilot recommended an asymmetric "−" dose (q10 − m) | **Not adopted:** symmetric centred ±(q90 − m), identical to the pilot, so the replication is exact. FVE guard applies per side. | Exact replication outranks dose cosmetics. |
| 57 | Projection arm, FollowText, split-injection | **Exploratory**, run only after §3.1–§3.5 are complete (D10 of the sprint). | Time and focus. |
| 58 | – | **Cut (D25):** next-digit cell; further template mining; the natural col arm beyond one line; the r11/q90 ambiguity; list-index control; backup steering. | D25. |
| 59 | Dry run on fresh items at n = 10 | **Dry runs only on the pilot split.** No fresh item is scored before the freeze manifest is recorded. | Forking paths. |
| 60 | n = 240 per feature from Hanley–McNeil | **n justified from pilot effect sizes and null-band decomposition (§8):** 240 fresh col passages (P1, P2, N, T1 subset), 160 fresh tense items (P3, P4, PC2), 120 fresh language items (PC1). The band, not n, limits power. | D15, §8. |

---

## 1. Research question, status, and what is being confirmed

Neel: *"NLAs seem to capture more than J-lens/Oracle Lens. Is that really the non-workspace stuff?"*

**Operational question.** At L42 (block-42 output = `hidden_states[43]`) of Qwen3.6-27B, with the text held fixed: does a reader's output follow a natural feature direction moved in its input activation, when that direction lies outside the top-r J-space (and the move is certified J-invisible by the leak test), compared against random directions of the same norm and against in-J features moved by the same relative norm?

Readers: NLA (`ceselder/qwen3.6-27b-nla-rl`, `av_rl_adapters/iter_000400`, L42 only), Oracle Lens (OL, s3d-rl600, L42 primary, L40/44 sensitivity), J-Lens (neuronpedia n1000; leak detector in the non-J arm, competitor in the in-J arm).

**Wording rule (unchanged).** "Outside the top-r J-space", never "outside the workspace". For P2/P3: "channel-specific sensitivity", never "reads".

### 1.1 Where the pilot left us (D15–D25, all on excluded pilot splits)
- **The one natural non-J feature that the model uses is col** (characters since the last newline): enrichment 0.67/1.10 at r\*/4r\*, J-Lens AUROC 0.500 [0.499, 0.501], NON-J under both lenses (D13), J-removed steering 49× random, keeps 79% of its causal effect after J-removal (tense 19%), not self-reportable (7% exact count).
- **Fixed col templates are flat for both decoders** at every direction and dose (NLA 0.47–0.53, OL 0.46–0.51, |z| ≤ 1.8), although at the same relative norm both decoders track raw language (NLA 0.87, OL 0.80) and the NLA tracks raw tense (0.85) (K3 PASS, D20).
- **Round 5 (D24) found two candidate narrow channels**, both leak-clean at 4r\* = 1536 under label and content sets:
  - NLA, one frozen descriptive template ("…near the end of a hard-wrapped line"): AUROC 0.63, z_shift +5.6 (iso) / +3.5 (Jc) at 4r\* q90 against the larger-norm proxy null; 0.66 at r11 (z +7.4 / +4.3). Per-template z values were recomputed for this document from the round-5 raw outputs (pilot split; one additional look, disclosed in §1.4). The other two f07_count templates: 0.52 and 0.57.
  - OL, newline continuation ("- \nthe" vs "- the"): 4r\* q90 AUROC 0.67, z +3.9 / +3.9 (proxy null); linear in dose; J-removal keeps 78% per unit dose vs the model's own 70%; **adapter-off collapses it** (0.40–0.44).
  - Every *descriptive* OL col template is flat or anti-tracking (0.46–0.50).
- **Tense 4r\* with frozen mined templates:** OL 0.66 (q90; z +12.5 / +16.6), NLA 0.59 (z +7.5 / +9.2). D20's "NLA > OL on tense 4r\*" came from OL's fixed label templates (0.52–0.53) and is **retracted**.
- **Behaviour at the primary col cell:** the 4r\* q90 edit shifts newline log-odds by +0.72 [0.62, 0.82] (z +55) at L42 and the L38 generation wrap by −1.55 [−2.7, −0.6] characters, while the J-Lens EOL readout stays at 0.50.
- **Pilot summary (D25):** "No robust descriptive reading; two *candidate* narrow channels, pre-registered for confirmation."

### 1.2 Retractions carried into this document
1. D21's working sentence "neither the NLA nor Oracle Lens reads it in any phrasing" — **retracted** (D24.1).
2. "NLA > OL on tense 4r\*" (D20; D21 item 6) — **retracted** (D24).
3. Rev-3 H2's "CI excludes 0.5" as a sufficient test — **retracted** (D15): random directions pass it.
4. Rev-3 "In-J count control = list index" — **dropped** (D19); "next-digit cell" — **dropped** (D24).
5. Rev-3 "WorkspaceBench success is fully explained by text" — **replaced** by the D22 wording (target information is prompt-recoverable).

### 1.3 Two-branch headline (D25), fixed now
Branches are keyed to the col primaries only (P1, P2); tense (P3, P4) is a geometric claim about the tail of an in-J feature that the model can already verbalise (87% reportable) and never carries the title.

| Branch | Condition | Abstract, first sentence (template) |
|---|---|---|
| **Branch 1** | P1 or P2 meets the full claim criterion (§3.1) | "Barely, in narrow channels, and not in the language they use to describe what they read": on the one natural non-J feature the model uses, [the NLA, via one pre-named phrasing / OL, via its next-token continuation channel] shows channel-specific sensitivity to a J-Lens-invisible, behaviourally active edit, while its descriptive templates (N1–N5) and free readouts (T1) do not express it. |
| **Branch 2** | Neither P1 nor P2 meets the criterion | "No, on the one natural non-J feature the model uses": at doses where both decoders track in-J features (PC1–PC2) and the edit moves the model's newline decision, neither decoder's output follows the J-removed col direction beyond random directions. |

**Stands on either branch:** Part A (what counts as non-J; the survival test is uninformative; the calibrated gate; the positional-bookkeeping survey, scoped); Part D (WorkspaceBench target information is prompt-recoverable, D22 wording); the latent-generated reconstruction (D16: the NLA rewrites the clue to fit a planted bridge, 12/12). Parts B (col) and C (the tense tail) take their wording from §10.

### 1.4 DISCLOSURE: the pilot's looks, and how the primaries were chosen
The four primary cells were **chosen after looking at pilot data**. Across three pilot rounds on excluded items (D15, D20, D24), we scored decoder cells as follows (counted from `08_col_pilot/RESULTS.md`; a "cell" = reader × direction × dose × template; family-level = the template-family mean):

| Round | Readers × directions × doses | Templates per decoder | Template-level cells | Family-level cells |
|---|---|---|---|---|
| Pilot (D15), col | {NLA@42, OL@42, OL@40, OL@44} × {raw, r\*, 4r\*} × {q90, q99} | 5 (3 families) | 120 centred (+120 additive) | 72 |
| Calibration (D20) | col raw r11/r14 × {NLA, OL@42}; lang, tense × 5 cells × {NLA, OL@42}; OL@40/44 raw q90 | 5 col; 3 lang/tense | 92 | 36 |
| Round 5 (D24) | col 4 directions × {q90, r11} × {NLA, OL}; OL adapter-off 6 cells; digit 6 cells; tense 4 cells | NLA 11 / OL 14 col; 3 digit; 6 tense | 302 | 134 |
| **Total** | | | **≈ 514 centred (≈ 634 with additive)** | **≈ 242** |
| of which col decoder cells | | | **≈ 340 (+18 adapter-off)** | **≈ 172** |

Each cell was evaluated with 2 statistics (shift, AUROC) against 2 bands, so ≈ 2,000 z-values were inspected. The col templates themselves came from a best-of-~7 selection on natural items (07; no steering data). For this revision we additionally recomputed per-template z and the null-SD decomposition for the eight cells in §8.1 from the round-5 raw outputs (pilot split, existing scores; no new scoring); this informed the n and K choices, not the choice of primaries.

**How each primary was chosen.** P1 is the single best of the 11 NLA col templates at the leak-free cell (the other two templates in its family gave 0.52 and 0.57). P2 is the only OL col family with signal (its 3 pairs are uniform, 0.66–0.67), fixed before round-5 scoring. P3/P4 are the tense 4r\* cells with the frozen f07_tense family, the templates under which the D20 reading reversed. q90 is the dose pre-declared as primary since revision 3 (row 28), not the dose with the larger pilot z (r11).

**Consequences we accept.** Under a normal band, one-sided p(z ≥ 3) = 0.00135, so among ≈ 340 correlated col cells ≈ 0.5 false z ≥ 3 cells are expected by chance alone; the pilot showed two clusters (NLA f07_count, OL nl_cont). The pilot effects are therefore treated as **hypotheses**, and expected to shrink on fresh items (winner's curse). No pilot number enters any confirmatory test; all primaries are re-estimated on fresh, passage-disjoint items with fresh null directions.

## 2. The attribution frame (intro schematic only)

| | **Reader tracks the activation** (output follows a feature moved with text fixed) | **Reader tracks the text** (output follows a text change with the feature clamped) |
|---|---|---|
| **Content in the J-space** | "Workspace reading" (positive controls PC1–PC2) | Re-solving / re-description of workspace content (Part D) |
| **Content outside the top-r J-space** | **Neel's hypothesis for the NLA** (P1–P2; P3–P4 for the tail of tense) | Re-solving / re-description of non-J content (natural col arm: text-visible, Gemini 0.97; exploratory) |

Indices (defined in §6): FollowFeature (paired shift and paired win, z vs null bands; cross-item AUROC secondary); fraction of probe = (AUROC − 0.5)/(AUROC_probe − 0.5), bounded by 1 in the steering arm; FollowText (exploratory in rev 4).

## 3. Confirmatory hypotheses (binding; in this order)

### 3.1 PRIMARY cells (Holm over 4)

Common definition for all four (§4.4 gives the construction): read-layer steering (arm R) at **L42**, read token as in the pilot (one in-line token per passage, index ≥ 8, after ≥ 1 newline, for col; the clause token after the finite verb for tense), **centred** construction h' = h − (v·h)v ± Δv, unit direction v = normalise((I − P_{S_1536}) w) where S_1536 = top-1536 right singular vectors of W_U·diag(1+w_norm)·J_42 (neuronpedia n1000), dose **Δ = q90 − m** of the natural projection on 02's bank (fixed constants, not estimated on fresh items), both candidates of a pair scored in one batch, LO = log p(+ candidate) − log p(− candidate), per-item effect e_i = (LO_i(+Δ) − LO_i(−Δ))/2, template-family LO = mean of the family's templates' LO.

| ID | Reader | Feature, direction | Dose (Δ; ‖Δ‖/median‖h‖) | Template(s), + vs − (frozen literally) | Pilot (excluded; proxy null) | Prior P(claim) |
|---|---|---|---|---|---|---|
| **P1** | NLA@42, scored right after `<explanation>\n` | col; 02 regression (covariance) direction `col_hs43.pt`, J-removed at 4r\* = 1536 (02 `jrem_r1536` export) | 7.098 (≈ 0.079) | `nla:nj1_col:m2` only: **"Mid-sentence continuation located near the end of a hard-wrapped line."** vs **"Mid-sentence continuation located at the beginning of a hard-wrapped line."** | AUROC 0.63; win 0.95; shift +0.332 (item SD 0.221); z +5.6 / +3.5 | 0.35 |
| **P2** | OL@42 (adapter on), candidate as the first `- ` bullet | same as P1 | 7.098 (≈ 0.079) | `NL_CONT_OL`, mean of 3: **"- \nthe" vs "- the"; "- \nand" vs "- and"; "- \nof" vs "- of"** | AUROC 0.67; win 0.98; shift +0.861 (SD 0.306); z +3.9 / +3.9 | 0.55 |
| **P3** | OL@42 | tense (+ = Past); 02 class-mean direction `tense_hs43.pt`, J-removed at 1536 | 5.228 (≈ 0.059) | f07_tense OL, mean of 3: **"- They were" vs "- They are"; "- The village had" vs "- The village has"; "- It was" vs "- It is"** (`ol:nj2_tense:m3, m2, m0`) | AUROC 0.66; win 1.00; shift +0.972 (SD 0.489); z +12.5 / +16.6 | 0.90 |
| **P4** | NLA@42 | same as P3 | 5.228 (≈ 0.059) | f07_tense NLA, mean of 3: **"Historical register" vs "Descriptive register"; "Historical factual" vs "Descriptive factual"; "Historical tone" vs "Descriptive tone"** (`nla:nj2_tense:m2, m0, m3`) | AUROC 0.59; win 0.94; shift +0.772 (SD 0.600); z +7.5 / +9.2 | 0.85 |

**Primary statistic.** z_shift = (ē_real − mean of null ē)/SD of null ē, computed separately against the isotropic band and the J-complement band (each K = 100 directions, same per-item norm and centred ± construction; §4.4). z_min = min(z_iso, z_Jc).

**Claim criterion (all four conditions):**
1. **Band:** z_min ≥ +3, AND the real ē exceeds all 100 null ē in each band (guards against non-normal band tails).
2. **Holm over P1–P4:** one-sided p_k = 1 − Φ(z_min,k); Holm step-down at family-wise α = 0.05. (Because z ≥ 3 ⇔ p ≤ 0.00135 < 0.05/4, Holm never binds when condition 1 holds; it is stated so that the family is explicit. The item-level test, ē's item-clustered 95% CI excluding 0, is also Holm-corrected and is necessary but not sufficient.)
3. **Sign agreement:** the unweighted mean of z_Jc over the cell's **declared-secondary family** (below; K = 20 bands at each member's matched norm) is > 0. Members voided by leak/FVE are dropped; if all are void, condition 3 fails.
4. **Validity:** the cell is not void under the leak test (§4.4.3) or the FVE guard (§4.4.4). A void primary counts as "not claimed"; it is never replaced, and Holm stays over 4.

A z ≤ −3 is reported as anti-tracking, never as a pass.

**Declared-secondary families (for condition 3; fixed now):**

| Primary | Family members (all at L42 unless stated; same direction) |
|---|---|
| P1 | (i) `nla:nj1_col:m3` ("Readout taken late/early in the span of the wrapped line.") at the P1 cell; (ii) `nla:nj1_col:m0` ("The token appears far into of / near the beginning of the wrapped line.") at the P1 cell; (iii) P1 template at r11; (iv) P1 template at q99 |
| P2 | (i) nl_cont at r11; (ii) nl_cont at q99; (iii) nl_cont at OL@40, 4r\* (r = 1536), q90 |
| P3 | (i) OL f07_tense at r11; (ii) at q99; (iii) at OL@40, r = 1536, q90 |
| P4 | (i) NLA fixed label tense family (3 templates, `r5_templates.NLA_PAIRS["tense"]["label"]`) at the P4 cell; (ii) f07_tense at r11; (iii) at q99 |

**Effect sizes reported for every primary (fresh items, item-clustered bootstrap, 2,000 resamples):** ē with 95% CI; d_z = ē/SD(e); paired win with 95% CI; cross-item AUROC with 95% CI; fraction of probe = 2(AUROC − 0.5) at probe AUROC ≈ 1.00 (probe re-measured on fresh items); z_iso, z_Jc and empirical percentiles; the behavioural shift at the same cell (§4.4.5).

### 3.2 Positive controls (dose-in-range evidence; K3)

| ID | Feature, direction | Dose | Templates | Readers | Pass rule | Prediction (pilot) |
|---|---|---|---|---|---|---|
| **PC1** | language (+ = Russian, − = Polish); 02 class-mean direction `lang_hs43.pt`, **raw** | **r11 = 0.11‖h_i‖ per item (matched relative norm)**; q90 reported | `calib_templates.py` fixed (3 per decoder: "Russian-language text" / "Polish-language text", etc.) | NLA@42, OL@42; J-Lens@42 as competitor | ≥ 1 decoder: z_min ≥ +3 AND AUROC ≥ 0.65 | NLA 0.87 (z +30), OL 0.80 (z +20) |
| **PC2** | tense (+ = Past); 02 class-mean direction, **raw** | r11; q90 reported | frozen f07_tense (same as P3/P4) | NLA, OL; J-Lens | same | NLA f07 0.92, OL f07 0.99 |

At the same matched norm r11, raw col is scored with the P1 template, nl_cont and N1–N5 (Figure 1; descriptive, not a control). **If PC1 and PC2 both fail → K3 fails → pivot P2 (§9).**

### 3.3 Pre-registered NEGATIVES (equivalence tests)

**Claim:** the OL *descriptive* col families do not express the J-removed col direction at the P1/P2 cell.

| ID | OL family at the P1/P2 cell (4r\*, q90, L42) | Templates (frozen) | Pilot AUROC (n = 60) |
|---|---|---|---|
| N1 | count_qual (fixed; `templates.py`) | "- near the end of a long line" / "- at the start of a line"; "- a long line of text" / "- a short line of text" | 0.49 |
| N2 | count_num (fixed) | "- about 60 characters" / "- about 20 characters" | 0.50 |
| N3 | decision (fixed) | "- line break coming up" / "- line break still far away"; "- next token is a newline" / "- next token is another word" | 0.49 |
| N4 | f07_count (frozen) | "- deep after a newline" / "- just after a newline"; "- line position: late in the line" / "…early…"; "- current line depth: deep" / "…shallow" | 0.49 |
| N5 | f07_break (frozen) | "- line wrap status: line break imminent" / "…mid-line, no break yet"; "- near linewrap boundary" / "- far from…"; "- line break is coming soon" / "…far away" | 0.46 |

- **SESOI:** cross-item AUROC 0.55, i.e. **fraction of probe 0.10** at probe AUROC 1.00 (if the fresh probe AUROC is A_p, the fraction SESOI is 0.05/(A_p − 0.5)); in band units, 0.05 AUROC ≈ 2–4 null-SD (OL AUROC null SD 0.012–0.029 in the pilot). Rev-3's "AUROC ≤ 0.55 = no meaningful tracking" is now the equivalence bound.
- **Test (TOST, per family):** item-clustered bootstrap 90% CI of AUROC ⊂ (0.45, 0.55) ⇔ both one-sided tests at α = 0.05. **Non-superiority** (upper 95% one-sided bound < 0.55) is the substantive claim ("does not express +col"); the lower bound distinguishes "flat" from "anti-tracking".
- **Family decision:** intersection-union — "OL descriptive templates are flat" is claimed only if all five pass TOST (no multiplicity correction needed for IUT). "OL descriptive templates do not express col" is claimed if all five pass non-superiority.
- **Prediction:** all five pass non-superiority. N5 may fail the lower TOST bound (pilot 0.46; at n = 240 SE ≈ 0.005–0.007), which is then reported as anti-tracking, with z vs both bands.
- **Declared NLA negatives (secondary, same test):** NLA count_qual, count_num, decision (fixed) and f07_break (frozen) at the P1 cell; pilot 0.49–0.53.

### 3.4 Template-free check (T1)

- **Question (verbatim, pre-written):** *"Does this text mention position within a line, line length, wrapping, or an upcoming line break? Answer yes or no."*
- **Judges (fixed):** Gemini 3.8 Flash (primary; OpenRouter, temperature 0, reasoning minimal, max_tokens 8, answer parsed as the first yes/no token) and `accounts/fireworks/models/kimi-k3` (second). The judge sees only the readout text.
- **Readouts:** free generations, k = 4 per item per condition (T = 1, top_p 0.95, top_k 64, 256 tokens; the readers' standard settings), on **120 fresh col passages** (a pre-drawn subset of the 240), for NLA@42 and OL@42, under 7 conditions: 0-dose; ±4r\* q90 (the P1/P2 cell); ±raw q90; ±raw q99. That is 120 × 4 × 7 × 2 = 6,720 readouts.
- **Judge sensitivity gate (before the sprint):** 40 synthetic readouts (20 that mention line position/length/wrapping/a break, 20 that mention "characters" in the narrative sense or nothing), written before any T1 readout exists. Both judges must classify ≥ 36/40 correctly; otherwise the prompt is revised once on the synthetic set only.
- **Statistics:** yes-rate per reader × condition with a Clopper–Pearson 95% upper bound (0/960 ⇒ ≤ 0.38%); item-level McNemar of any-yes on +Δ vs −Δ; judge agreement (κ, and raw agreement since κ is degenerate near 0). Every "yes" by either judge is shown to the user, who labels it blind to condition.
- **Prediction:** upper bound < 1% for every reader × condition, and no + vs − difference. **Falsified** if a confirmed (user-labelled) yes-rate exceeds 1% in a steered condition and exceeds the opposite sign's rate (McNemar p < 0.05).

### 3.5 Declared secondaries (reported with their own tests; not part of the Holm family)

| ID | What | Test / prediction |
|---|---|---|
| S1 | Class-mean col direction (long ≥ 45 minus short ≤ 20 on 02's wrap bank; cm_jrem4) at the P1/P2 cell | Same statistics. Not an independent replication (cos(cm_jrem4, reg_jrem4) = 0.996); predicted to agree with P1/P2 within ±0.5 z. |
| S2 | r11 dose for P1–P4 (sensitivity) | Reported beside q90; never replaces it. |
| S3 | q99 dose for P1–P4 (FVE-guarded per side) | Dose monotonicity (old H6): ē(q99) > ē(q90) where both are valid. |
| S4 | Layers: OL@40 and OL@44 for P2/P3 with r = 1536 at both (not 4r\*(L44) = 1024, which leaked EOL in the pilot); leak-tested at the cell's own layer | Reported; L44 void if its leak fires. |
| S5 | Additive steering (h ± Δv) for P1–P4 | Qualitatively equal to centred (pilot: 0.54 vs 0.52). |
| S6 | **OL mechanism (D25):** (a) indentation (non-J by G2, causally inert by G4) steered in OL's carrier with an indentation continuation pair; (b) gain ratio = OL continuation shift / model behavioural shift (same item, same edit) for col 4r\*, col raw, tense 4r\*, tense raw and 2 random directions with nonzero behavioural effect; (c) tense adapter-off (base Qwen in OL's carrier, P3 templates) | Re-instatement predicts: (a) flat; (b) ratios within ±25% of their mean across directions; (c) P3 effect collapses (as col: 0.40–0.44). "Reporting beyond acting" predicts (a) a shift. If `experiments/09` runs on the pilot split before the freeze, its numbers are cited in §14 and the predictions above are not changed. |
| S7 | Behavioural validation for P1–P4 in arm U (L38 upstream edit, centred, q90): newline log-odds, generation wrap shift (greedy, 48 tokens); tense next-verb log-odds | Reported; pilot col 4r\* L38: +0.33 log-odds (z +45), −1.55 chars. |
| S8 | (C) planted intermediates as **in-J calibration**, α ≤ 1.5 subset at L38 (109 items in the yield check); (S8b) the **latent-generated reconstruction rate**: fraction of free NLA readouts on planted items whose reconstructed prompt is rewritten to fit the planted bridge (judged by both judges, user-checked on 30) | (C): all readers report the planted bridge (pilot ≥ 0.98). S8b: ≥ 0.7 (pilot 12/12). |
| S9 | NLA negatives (§3.3) | TOST as for N1–N5. |

### 3.6 Parts that stand on either branch (confirmatory replications of completed results)
- **Part A, HA1 (survival test uninformative):** on held-out items, for every gated feature, probe retention under J-removal minus retention under random equal-rank removal ∈ [−0.02, +0.02] at r\* and 4r\*, under neuronpedia n1000 and camilablank (D10, D13). Prior 0.85.
- **Part A, HA2 (calibrated gate):** col NON-J under both lenses; tense, language, POS (token-class functional gate) IN-J. Prior 0.8.
- **Part A, survey:** the 16 gate runs (02: col, col_report, depth, pred_pos, tense, lang, topic; 04: caps, code, quote, gender, indent, wordpos, list index ×2, plus 08's digit), reported as "on Qwen3.6-27B at L42, the J-depleted, causally used content among ≈ 13 surveyed features is positional bookkeeping" (D19, D21), with "a survey, not a theorem" in the same sentence.
- **Part D (no new data):** D22 wording verbatim. The WorkspaceBench floor issue (D17) is reported only if the user approves filing it.
- **Latent-generated reconstruction:** S8b.

### 3.7 Exploratory (labelled "exploratory" wherever reported)
Everything not in §3.1–§3.6, including: NLA-vs-OL differences on any cell (old H3); per-template breakdowns beyond the declared families; the natural col arm (one line, next to Gemini 0.97 and the Gemini-regressed residual 0.51/0.47); FollowText with clamp validation; the Viswanath projection arm; NLA iter 300; OL best-layer curves; L38 arm U reader scoring; any new template, direction or feature.

## 4. Operational definitions

### 4.1 J-space
- S_r = top-r right singular vectors of M = W_U·diag(1 + w_norm)·J_L (neuronpedia `jacobian-lens` n1000 wikitext). **r\* = 384, 4r\* = 1536** (readout-agreement rule, D10). At L44 the secondaries use r = 1536 (row S4).
- Enrichment = (energy in S_r)/(r/d), d = 5120; always reported with a shuffled-label reference (D19: shuffled labels give 1.2–2.5× at r\*).
- camilablank lens: cross-check only (HA1/HA2).
- **No rank makes a direction J-invisible; the leak test certifies a dose** (D9.3).

### 4.2 Feature gates (unchanged from rev 3; final labels)
G1 decodable; G2a enrichment of the diff-in-means direction ≤ 1.5 at r\* and 4r\*; G2b the J-Lens readout does not contain the value in the form the lens emits (shuffled-label band); G4 causal use along the J-removed direction. Labels NON-J / DIFFUSE-BUT-J-READABLE / IN-J / decodable-unused.

| Feature | Label (final) | Key numbers |
|---|---|---|
| col | **NON-J** (both lenses); "count non-J; the decision it feeds is weakly J-visible" (soft newline mass AUROC 0.58) | R² 0.655; enrichment 0.67/1.10 (camilablank 0.82/1.41); J-Lens 0.500; 49× random; keeps 79% |
| tense | IN-J | enrichment 5.6/2.3; J-Lens 0.52; keeps 19% |
| language (ru/pl) | IN-J | enrichment 3.3/1.7; J-Lens 0.62 |
| pred_pos | IN-J (token-class functional gate 0.66–0.69) | enrichment 2.62/1.56 |
| col_report | NON-J (mini-finding: explicit report did not move the count into J) | enrichment 0.59/0.86 |
| depth, code, wordpos | DIFFUSE-BUT-J-READABLE | – |
| caps, quote, topic | IN-J | – |
| indentation | decodable, unused (G4 fails) | enrichment 1.11/1.14 |
| gender | fails G1 | AUROC 0.60 |
| list index, next-digit | failed as in-J count controls | not enriched vs shuffled labels; digit 0/30 flips |

### 4.3 Feature set for the main run
- **Non-J:** col only (P1, P2, N1–N5, T1, S1–S7).
- **In-J feature whose J-removed tail is tested:** tense (P3, P4).
- **Positive controls:** language (PC1), tense raw (PC2).
- Cut: backups, list index, next-digit (row 58).

### 4.4 Steering, nulls, leak, FVE, behaviour

#### 4.4.1 Construction and doses
- **Arm R (all confirmatory cells):** the edited vector is handed to the reader at L42 (OL@40/44 for S4); no model forward pass after the edit.
- **Centred:** h' = h − (v·h)v ± Δv. Additive is S5. Norm-preserving steering is not used (no-op for both decoders, D11).
- **Doses (fixed constants from 02's banks; recorded in the freeze manifest):** col reg_raw q90 9.465 / q99 13.999; col reg_jrem4 q90 7.098 / q99 10.819; col cm_jrem4 q90 7.138; tense raw q90 8.392 / q99 13.904; tense jrem4 q90 5.228 / q99 9.100; r11 = 0.11‖h_i‖ per item. Symmetric ±Δ (row 56).
- **Directions:** col = 02 regression direction (P1/P2), cm (S1); tense, language = 02 class-mean directions. J-removed = 02's `jrem_r1536` exports at L42 (cosine to our recomputation 0.99999).

#### 4.4.2 Null bands
- For every primary: **K = 100 isotropic and K = 100 J-complement (orthogonal to S_1536) unit directions**, drawn with a fresh seed recorded in the manifest (not the pilot's 20), scaled to the **exact per-item norm of the real cell** (q90 constants for P1–P4; r11 for PCs), same centred ± construction, same items and templates. Secondaries: K = 20 per class at their own matched norm.
- Reported as z and empirical percentile; **never subtracted**.

#### 4.4.3 Leak test (two-sided; certifies non-J doses)
- J-Lens@L (L = the cell's read layer; L42 for primaries, L40 also reported), max cosine over each token set, ± sides, statistic = mean shift and AUROC, against the K = 100 J-complement band.
- **Primary sets (fire at the central 95% band, either direction):**
  - col: EOL (all 200 newline-containing tokens; all whitespace-only in Qwen's vocabulary); count label words (long/longer vs short/shorter; end/ending vs start/beginning; sixty/seventy/eighty vs ten/fifteen/twenty); content3 (the/and/of, 12 variants).
  - tense: label (past/was/were vs present/is/are); content (r5 10 vs 10 verbs: was, were, had, did, said, became, went, made, took, came vs is, are, has, does, says, becomes, goes, makes, takes, comes) ∪ the calibration's 34 vs 32 verb-token set (`calib_modal.jlens_content`).
  - language (PCs; reported only, raw is in-J by construction): label + Cyrillic vs Polish-diacritic content.
- **Supplementary sets (fire only at |z| ≥ 3):** content10; EOL − content3; EOL − content10.
- **Rule:** if any primary set fires, the dose is **void for non-J claims** (P1–P4 count as not claimed). The expected false-void rate is ≤ 1 − 0.95³ ≈ 14% per col cell if sets were independent (they are positively correlated, so lower). We accept this conservatism.
- Pilot: all P1–P4 cells clean (col 4r\* EOL z +0.2, content3 −0.1; tense 4r\* label +0.4, content −0.9; Jc band).

#### 4.4.4 FVE guard (one-sided)
- NLA AV + AR, k = 4, per-example paired natural vs steered, per side (+Δ and −Δ separately). **Void** if mean FVE(steered) < mean FVE(natural) − 0.05. The guard is a property of the edited vector, so it applies to the OL cell with the same direction and dose too.
- High FVE proves nothing; the in-range evidence is PC1–PC2 at matched relative norm.
- Pilot: col 4r\* q90 −0.010 / −0.018 (valid). Tense 4r\* q90 not yet measured (K2).

#### 4.4.5 Behavioural validation at the primary cell (reported; a precondition for wording)
- **col (P1/P2 cell):** L42 read-layer edit at the read token, newline log-odds at the next position, e = (plus − minus)/2, z vs a K = 20 J-complement band; L38 centred edit, greedy 48-token generation, plus-minus characters to the first newline. Pilot: +0.72 [0.62, 0.82] nats (z +55); −1.55 [−2.7, −0.6] chars.
- **tense (P3/P4 cell):** L42 edit, log P(past verb set) − log P(present verb set) at the next position, using 02's tense G4 metric and the content set above.
- **Rule:** if the behavioural shift at a primary cell is not ≥ +3 SD of its band, a passing primary is worded "reader sensitivity to a behaviourally inactive direction" (the digit cell showed decoders track inert directions, D24) and cannot carry Branch 1.

### 4.5 (C) planted intermediates — in-J calibration (S8)
As in revision 3 §4.5 and `experiments/05`, with the centred clamp swap at L38, keep criteria (i) flip, (ii-a) neutral KL ≤ 0.1, (ii-b) prompt still decodable at L42, (iii) the X-vs-Y probe reads Y. **Primary subset α ≤ 1.5** (109 items; ‖Δ‖/‖h‖ well above the natural X–Y distance, stated). Reported as in-J calibration, never as a re-solving test (D21).

### 4.6 Projection arm, FollowText — exploratory (§3.7)
Definitions unchanged from revision 3 (§4.4 FollowText with clamp validation ≤ 0.55, INLP k ≤ 8; §4.6 projection arm). Run only on sprint D10 if §3.1–§3.6 are complete.

## 5. Readers, layers, matched comparisons

| Reader | Layer(s) | Role here | Notes |
|---|---|---|---|
| NLA `ceselder/qwen3.6-27b-nla-rl`, `iter_000400` | 42 | P1, P4, PCs, T1, negatives (S9) | iter 300 exploratory. Candidate after `<explanation>\n`. |
| OL s3d-rl600 | 42 (primary); 40, 44 (S4) | P2, P3, PCs, N1–N5, T1, S6 | Candidate as first `- ` bullet; capture with adapter disabled; injection `16000·unit(h)`. |
| J-Lens neuronpedia n1000 | 40, 42, 44 | Leak detector (non-J); competitor (PCs) | Max cosine over single-token variants. |
| Linear probe (02 ridge / class-mean) | 42 | Fraction-of-probe denominator | ≈ 1.00 in the steering arm by construction. |
| Judges | – | T1, S8b | Gemini 3.8 Flash; kimi-k3. |

All readers get identical captured vectors (capture once, block-42 output). DeltaNet kernels pinned as in 03.

## 6. Scoring and statistics

### 6.1 Paired log-odds, frozen literally
- LO_R(h) = log p_R(T(+) | prompt_R, h) − log p_R(T(−) | prompt_R, h), teacher-forced; both candidates of a pair in **one batch** (batch composition shifts scores by up to 0.26–0.64 nats, D5, R5.2d).
- **Template files and hashes (verified at K5 before any scoring):**
  - `experiments/07_template_mining/templates_frozen.json` sha256 `09156722b059edb99d01e919c29b2ff660e37cba3152f2d3ce67f3272d3e923b`;
  - `experiments/08_col_pilot/r5_templates.py` sha256 `40d18f289b9e3ae1fbb0c5d1207280ab170c2f54e318e451b245fa7d06fd3ff5` (verbatim copy of 07's selected templates + `NL_CONT_OL`);
  - `experiments/08_col_pilot/templates.py` sha256 `4de335840d27bcc0c47bb2bb835582f4beb77d2c412ba6df4269af381464afb1` (fixed col families);
  - `experiments/08_col_pilot/calib_templates.py` sha256 `ce6d758dc774f05f201d37ca1b407b181592bb10d26d8990698ede376e3b488f` (language/tense fixed).
- **No further mining, no new families, no edits** (typos included). Sign convention: + = longer line / Past / Russian.

### 6.2 Statistics
- Unit = passage (col: one read position per passage, as piloted; the multi-position ICC design of rev 3 is not used). Item-clustered bootstrap, 2,000 resamples, for all CIs.
- Primary: z_shift vs both bands (K = 100) with empirical percentiles; Holm over P1–P4; sign agreement (§3.1). Secondary per cell: z_AUROC, paired win, cross-item AUROC, fraction of probe.
- Negatives: TOST on cross-item AUROC (§3.3).
- T1: Clopper–Pearson, McNemar (§3.4).
- **Analysis code frozen** (`experiments/10_main/main_analyze.py` [DECIDE: path]) and tested on the pilot split, where it must reproduce R5's published numbers to ±0.01 AUROC / ±0.1 z (with the pilot's proxy nulls). Its sha256 enters the freeze manifest. The primary analysis is one run of this script; its stdout is saved verbatim.
- **Deviations:** any change after the freeze is logged in `notes/deviations.md` with time, reason and whether data had been seen; results from deviated analyses are labelled exploratory.

## 7. Controls (summary)
0-dose; K = 100 isotropic and J-complement null bands at matched norm (K = 20 for secondaries); positive controls PC1–PC2 at matched relative norm; leak test with label + content sets; FVE one-sided guard; behavioural validation at each primary cell; adapter-off (S6c) and gain-ratio (S6b) mechanism checks; judge sensitivity set (T1); template and code hashes; item disjointness (K5); the Gemini text ceiling for any natural col number (D23).

## 8. Sample sizes and power

### 8.1 The band, not n, limits power
The SD of the null-band mean shift decomposes into direction-to-direction variance (fixed) and item noise (∝ 1/√n): SD(n) = √(σ_dir² + σ_item²/n). Recomputed from the round-5 raw outputs at the exact primary cells (pilot proxy nulls; per-template for P1):

| Cell | ē (item SD) | Band | σ_dir | σ_item/√60 | z at n = 60 / 120 / 240 / ∞ |
|---|---|---|---|---|---|
| P1 NLA m2, col 4r\* q90 | +0.332 (0.221) | iso | 0.051 | 0.031 | +5.6 / +6.0 / +6.3 / +6.5 |
| | | **Jc** | **0.088** | 0.030 | **+3.5 / +3.6 / +3.6 / +3.7** |
| P2 OL nl_cont | +0.861 (0.306) | iso | 0.213 | 0.033 | +3.9 / +4.0 / +4.0 / +4.0 |
| | | Jc | 0.207 | 0.031 | +3.9 / +4.0 / +4.0 / +4.0 |
| P3 OL f07_tense, tense 4r\* q90 | +0.972 (0.489) | iso / Jc | 0.077 / 0.058 | 0.008 / 0.007 | +12.5 / +16.6 at n = 120 (asymptote +12.6 / +16.8) |
| P4 NLA f07_tense | +0.772 (0.600) | iso / Jc | 0.102 / 0.082 | 0.017 / 0.016 | +7.5 / +9.2 at n = 120 (asymptote +7.7 / +9.5) |

- **Reading.** Going from 60 to 240 passages raises P1's binding (Jc) z by ≈ 0.1 and P2's by ≈ 0.05. No feasible n moves a marginal cell across z = 3; only the true effect does. **P1 and P2 sit 0.5–1.0 SD above the bar in the pilot, before any winner's-curse shrinkage**: P1's family-mean z is only +2.5–2.9, so a regression of P1 toward its family would fail it. The priors in §3.1 reflect this.
- **What n buys:** (a) precise effect sizes: at n = 240, SE(ē) ≈ 0.014 (P1) and 0.020 (P2); SE(AUROC) ≈ 0.01; d_z CIs ±0.15; (b) TOST power for N1–N5: pilot SE(AUROC) ≈ 0.010 at n = 60 → ≈ 0.005 at n = 240, so a true AUROC of 0.49 ± 0.03 passes TOST with power > 0.99; (c) T1's upper bound (0/960 ⇒ 0.38%).
- **K matters more than n:** with K = 20 the band SD has relative SE ≈ 1/√(2·19) ≈ 16% (z = 3.5 ± 0.55); with K = 100, ≈ 7% (± 0.25). Hence K = 100 for primaries.

### 8.2 Sample sizes (fresh, passage-disjoint items)
| Bank | n | Used for | Justification |
|---|---|---|---|
| col passages (02 `build_wrap` recipe, W ~ U{30..90} as 07; one read token per passage, index ≥ 8, after ≥ 1 newline) | **240** | P1, P2, N1–N5, S1–S7, behaviour; **120 pre-drawn** for T1 | z saturates by 120 (≥ 97% of asymptote); 240 for effect-size CIs and TOST; matches D21.8. |
| tense items (02 `build_prose`, spaCy clause tense, 80 Past / 80 Present, one per passage) | **160** | P3, P4, PC2, S6c | Effects ≥ 7 SD at n = 120; 160 gives balanced classes and SE(AUROC) ≈ 0.02 for the NLA-vs-OL exploratory contrast (D21: "more OL tense items"). |
| language items (papluca **test** split, 60 ru + 60 pl, disjoint from calibration_items) | **120** | PC1 | Pilot d_z ≈ 1.2–1.5 at n = 60. |
| (C) planted | existing α ≤ 1.5 subset (109 items) | S8 | yield check (D16). |

**Disjointness (K5):** zero overlap with `pilot_split.json` (sha256 `42643350…f902`; including `calibration_items` and `round5_items`), with every id in `templates_frozen.json` `items[*].mine|select`, and with 02's wrap/prose pickles; passage-level sha1 of the whitespace-normalised paragraph plus its 80-char head, as in the pilot.

### 8.3 Compute
Per primary: 240 (or 160) items × (2 real + 400 null conditions + secondaries) × candidates. NLA ≈ 0.23 s per activation (10 candidates/batch), OL ≈ 0.12 s. Estimate: NLA ≈ 8 H200-h, OL ≈ 5 H100-h for primaries + PCs + negatives; generations for T1 (6,720 readouts) ≈ 2 GPU-h; behaviour ≈ 1 GPU-h; secondaries ≈ 6 GPU-h. Total ≈ 25 GPU-h (≈ $110) plus judges < $10.

## 9. Kill checks and pivots

| When | Check | Status / action |
|---|---|---|
| K0 | NLA layer convention (functional) | **PASSED** (D5): block-42 output, FVE 0.760. |
| K1 | ≥ 1 non-J feature passes G1–G4 | **PASSED with n = 1** (col; D10, D13, D19). Frozen: no backup search. |
| K4 | (C) yield ≥ 60 | **PASSED** (431 at L38; 109 at α ≤ 1.5). |
| **K5** (sprint D1, before any scoring) | Integrity: template and analysis-code hashes match the manifest; item disjointness = 0 overlaps; K = 100 matched-norm nulls exist for all four primaries and both PCs | Fail → fix before scoring; log in `deviations.md`. No reader score is computed until K5 passes. |
| **K2** (D1–D2) | At each primary cell: FVE guard (both sides), leak test (primary sets), behavioural validation | FVE or leak fails → that primary is **void** (counts as not claimed; Holm stays over 4). Behaviour fails → §4.4.5 wording. All of P1/P2 void → Branch 2 by default, worded "no valid non-J test". |
| **K3** (D3) | PC1 and PC2 on fresh items at r11 | Both fail → readouts or doses broken → **pivot P2**: Part A + Part D + latent-generated reconstruction, col reported as "untestable". One fails → proceed; state which control holds. |
| K6 (D3) | Band shape: Shapiro–Wilk on the 100 null shifts per band | p < 0.01 → the empirical-percentile condition (exceeds all 100) is decisive; z reported as descriptive. |

**Pivots.** P1 (old) is retired: a flat col with passing PCs is Branch 2, a result. P2 as above.

## 10. Decisive outcomes (fixed wording)

| P1 (NLA col) | P2 (OL col) | N1–N5 | T1 | Branch / Part B wording |
|---|---|---|---|---|
| claim | claim | pass (non-superiority) | ≈ 0 | **Branch 1:** both decoders carry a narrow, J-Lens-invisible channel for col (one NLA phrasing; OL's continuation channel), absent from their descriptive templates and free readouts. |
| fail | claim | pass | ≈ 0 | **Branch 1 (OL only):** "OL's continuation channel carries the model's newline decision, including its J-Lens-invisible part; OL does not describe it." Mechanism wording from S6 (re-instatement if S6a flat and S6b ratios constant). |
| claim | fail | pass | ≈ 0 | **Branch 1 (NLA only):** "one pre-named NLA phrasing tracks the J-removed col edit; other phrasings and free readouts do not." n = 1 template stated. |
| fail | fail | pass | ≈ 0 | **Branch 2:** "No, on the one natural non-J feature the model uses" (PCs pass at matched norm; behaviour moves). |
| any claim | – | ≥ 1 family fails non-superiority | – | Descriptive language does carry col: Branch 1 wording drops "not in the language they use to describe"; reported as a pre-registered negative that failed. |
| – | – | – | confirmed yes-rate > 1%, + > − | Template-free reading of col exists; strongest Branch 1 form; reported as a failed prediction. |
| claim only at r11 (S2), fail at q90 | | | | **Not claimed.** Reported as a sensitivity result. |
| claim, behaviour fails at the cell | | | | "Reader sensitivity to a behaviourally inactive direction"; cannot carry Branch 1. |

| P3 (OL tense 4r\*) | P4 (NLA tense 4r\*) | Part C wording |
|---|---|---|
| claim | claim | Both decoders follow the J-Lens-invisible tail of tense (beyond the top-1536 J-space), a feature the model can verbalise (geometric claim, not "reads what the model can't say"). NLA-vs-OL comparison exploratory. If S6c collapses OL's effect, OL's version is worded as continuation-channel re-instatement. |
| claim | fail | The tail is carried by OL's continuation channel only. |
| fail | claim | The NLA follows the tail; OL does not (the D20 pattern, now confirmatory). |
| fail | fail | Tense 4r\* moves to the appendix (D21). |

## 11. Figures

**Figure 1 (D21).** Three readers (NLA, OL, J-Lens) × {language, tense, col} at **matched relative norm** (r11 = 0.11‖h‖; q90 values marked), raw and 4r\* directions side by side; y = z_shift with **both null bands** drawn (iso grey, J-complement dark grey; ±3 lines). For col, separate points for the P1 template, nl_cont, and N1–N5 (descriptive families). J-Lens drawn as the leak trace in non-J cells; void cells hollow. Inset strip: behavioural shift at the same cells.

```
            language (in-J)      tense (in-J)          col (non-J)
 z vs band  raw   4r*            raw   4r*             raw        4r*
   +30 ┤  N●  O●                N●  O●
   +10 ┤        N○void         ...    O● N●
    +3 ┼ - - - - - - - - - - - - - - - - - - - - - - P1? P2?  - - - - -
     0 ┼                                            desc ●●●   J-Lens ○
    -3 ┼ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

**Figure 2 (primaries).** Forest plot of P1–P4 on fresh items: ē with 95% CI, z_iso and z_Jc with the K = 100 band histograms, pilot estimate as a hollow marker (shows shrinkage), declared-secondary family members as small markers.

**Figure 3 (negatives and T1).** N1–N5 (and NLA S9) AUROC with 90% CI against the (0.45, 0.55) equivalence region; T1 yes-rates with Clopper–Pearson bounds per reader × condition.

**Figure 4 (Part A).** Survival test vs random equal-rank removal (all ≈ 1) beside the calibrated gate (enrichment at r\*/4r\*, G2b z), for the ≈ 13 surveyed features, labelled NON-J / DIFFUSE / IN-J / unused.

**Supplementary.** Dose diagnostics (‖Δ‖/‖h‖, FVE per side, KL); leak traces per set and layer; S1–S7; (C) and S8b; Part D table (D22); exploratory panels labelled.

## 12. Timeline

**Prep (Oct 7–18).** No fresh item is scored before the freeze.

| Date | Work |
|---|---|
| Oct 7–8 | `experiments/09`: OL mechanism tests on the pilot split (S6 a–c), if run; matched-norm null generator (K = 100, seed recorded); per-side FVE for tense 4r\* on the pilot split. |
| Oct 9–10 | Build fresh banks (240 col, 160 tense, 120 language); disjointness checks; record bank sha256s. Write the 40-item judge sensitivity set; run both judges on it only. |
| Oct 11–13 | Write `main_analyze.py`; reproduce R5 on the pilot split (±0.01 AUROC, ±0.1 z); unit tests for Holm, TOST, band z, sign agreement. |
| Oct 14–15 | Dry run of every arm on the pilot split (n = 10); compute and cost check. |
| Oct 16–17 | Fable review of rev 4; user approval; resolve §14. |
| Oct 18 | **Freeze v1**: manifest of sha256s (this file, the four template files, `pilot_split.json`, bank manifests, `main_analyze.py`, null seeds, dose constants) sent to the user and Fable with a timestamp (no git commits in this repo). |

**Sprint (Mon Oct 19 – Sun Nov 1).**

| Day | Date | Work | Gate |
|---|---|---|---|
| D1 | Mon Oct 19 | K5 integrity; capture L40/42/44 for all fresh banks; FVE guard, leak test and behavioural validation at P1–P4 cells | **K5**, K2 (part) |
| D2 | Tue Oct 20 | Score P1–P4 (real + K = 100 × 2 bands) and PC1–PC2; behaviour at L38 (S7) | **K2** |
| D3 | Wed Oct 21 | One run of `main_analyze.py` on primaries, PCs and N1–N5; K3, K6; record the branch; no re-runs except logged bug fixes | **K3**, K6 |
| D4 | Thu Oct 22 | T1: 6,720 generations; both judges; user labels every "yes" blind to condition | |
| D5 | Fri Oct 23 | Declared-secondary families (K = 20) → sign-agreement condition; S1–S5 | claim criterion final |
| D6 | Sat Oct 24 | S6 OL mechanism tests on fresh items (indentation, gain ratio, tense adapter-off) | |
| D7 | Sun Oct 25 | One-page interim (branch, P1–P4 table, Figure 1 draft) to Fable/Neel | |
| D8 | Mon Oct 26 | (C) α ≤ 1.5 in-J calibration; S8b reconstruction rate (judges + user check of 30) | |
| D9 | Tue Oct 27 | Part A replications (HA1/HA2, both lenses) on held-out items | |
| D10 | Wed Oct 28 | Exploratory: natural col one-liner with Gemini ceiling, FollowText, iter 300, projection arm (≤ half a day) | |
| D11 | Thu Oct 29 | Figures 1–4 and supplementary | |
| D12 | Fri Oct 30 | Statistics audit; deviations log; exploratory labels checked | |
| D13 | Sat Oct 31 | Write-up: abstract first sentence from §1.3; pilot and Viswanath in paragraph 1; disclosure §1.4 verbatim in the methods | |
| D14 | Sun Nov 1 | Buffer; red-team with Fable | |

## 13. Risks
1. **Forking paths (main threat).** ≈ 340 col cells looked at; P1–P4 chosen from them (§1.4). Mitigations: one named template/family per primary; fresh items and fresh null directions; K = 100; Holm over 4; z ≥ 3 vs both bands plus "exceeds all 100"; sign agreement with a declared family; disclosure in the paper. Residual: the declared families share items and direction with their primary, so sign agreement guards against template flukes, not against direction-specific artefacts.
2. **Single template (P1).** P1 is one sentence. A pass does not show the NLA "describes line position" (its other phrasings and T1 decide that); a fail does not show the NLA cannot. Winner's curse: P1 was best of 11 NLA col templates; its family mean was z +2.5–2.9. Expect shrinkage; prior 0.35.
3. **Re-instatement (P2, P3).** OL is Qwen + LoRA; the 78% vs 70% gain match favours the LoRA re-instating the activation so Qwen's own next-token circuitry acts on it. A pass in P2/P3 is then "OL inherits the model's decision", usable only if one already knows to measure the continuation. S6 tests this; the wording in §10 is fixed now. The NLA has no adapter-off analogue (av_base is fully fine-tuned).
4. **Band-limited power.** P1/P2 are within ≈ 1 SD of the bar in the pilot; n cannot rescue them. Accepted; stated in the paper.
5. **Leak-test conservatism.** Three primary col sets at the 95% band can void a valid cell (≤ 14%). Accepted; void cells are reported, never re-tested with fewer sets.
6. **Count-calibration gap.** No in-J count feature exists on Qwen (list index and digit failed). PCs are language and tense; the col null (if Branch 2) is scoped to "this feature", not "counts".
7. **n = 1 non-J feature.** Stated in Part B's first sentence; the survey (Part A) is the scope statement.
8. **Item and recipe drift.** Fresh banks use the pilot generators; any generator change is a deviation.
9. **Judge validity (T1, S8b).** Sensitivity gate on synthetic readouts; user labels every "yes"; Gemini over-credits on WorkspaceBench (D14: 0.43 vs 0.33), so T1 "yes" is never accepted without the user label.
10. **Dose skew.** Symmetric −q90 puts col's projection below its natural 1st percentile (FVE "−" side −0.018 at 4r\* q90, valid); the per-side FVE guard covers it.

## 14. Open decisions

**Resolved in revision 4 (from D20–D25):** confirmatory structure and order; P1–P4 definitions; q90 primary, r11 sensitivity; Holm over 4 + |z| ≥ 3 vs both bands + sign agreement; K = 100 for primaries; label + content leak sets; SESOI AUROC 0.55 (fraction 0.10), TOST; T1 question and judges; templates frozen literally; disclosure; two-branch headline; (C) as in-J calibration at α ≤ 1.5; retractions; cut list.

**Resolved earlier (unchanged):** r\* = 384 / 4r\* = 1536; neuronpedia n1000 primary; final G2; centred steering; L42 primary for all readers; NLA iter 400; FVE margin 0.05 per side, k = 4; kimi-k3 second judge; Gemini 3.8 Flash; sprint dates; no outreach to the pilot author.

**Still open (to close by Oct 17):**
1. **[DECIDE: Fable]** Leak-test rule for the three primary col sets: 95% band (as written; conservative) vs |z| ≥ 3 for all sets.
2. ~~[DECIDE: lead]~~ **Resolved:** frozen analysis script under `experiments/10_main/` (its hash recorded at freeze); deviations log `notes/deviations.md` (every departure, dated, with reason).
3. **[DECIDE: user]** Who writes the 40-item T1 judge sensitivity set (default: the lead, before any T1 readout exists).
4. **[DECIDE: Fable]** Whether `experiments/09` (S6 on the pilot split) must finish before the freeze, or S6 runs only on fresh items in the sprint (default: run on the pilot split if done by Oct 15; predictions in S6 are not changed by it).
5. **[DECIDE: user]** Filing the WorkspaceBench floor issue (D17, D21); Part D reports it only after approval.
6. ~~[DECIDE: lead]~~ **Resolved:** the union of the r5 10/10 list and the calibration 34/32 list (as written; the broader set is the more conservative leak test).


---

## 15. Re-analysis of the concurrent pilot (descriptive; ≤1 page in the write-up)

All numbers are recomputed from the raw JSONL of `LittleMeHere/nla-vs-lenses` (commit `58d34a8`); details are in `pilot_deep_dive.md`. This analysis is descriptive and motivates the design. It is not a test of our hypotheses.

1. **The NLA finds no hidden step that OL misses.** Multihop at L42 (n = 100): the NLA names the bridge where OL misses on **3** items; OL names it where the NLA misses on **8** (McNemar p = 0.23).
2. **The two prose readers succeed on the same items.** phi(OL, NLA) = **0.69**, vs phi(J-Lens, OL) = 0.35 and phi(J-Lens, NLA) = 0.32. That fits both decoding the same prompt-recoverable content.
3. **Bridge hits are solved-fact statements.** The bridge is named together with the answer in **67/75** NLA hits and **79/80** OL hits. The model's own top-1 next token is already the answer on 74/100 items.
4. **Single-sample noise is large.** Readouts on the original vs a near-identity random-direction control flip on 7–10 of 50 items (Cohen's κ = **0.55–0.70**). Differences under about 10/50 are inside this noise, which covers the swap and brew results.
5. **The swap test was weak.** The planting scale escalated until the J-Lens showed the planted bridge, and **34/50 items hit the 16× cap**. The J-Lens shows the planted bridge in only 14–17/50. OL and the NLA follow it in 6/17 and 4/17 of those items.
6. **Survival of deletion is ambiguous.** After deleting about 31 J-Lens directions, 52–59% of the prose readers' bridge mentions survive. Re-solving from the prompt predicts this, and so does reading bridge information outside the top-r J-space. Our FollowFeature (text fixed) vs FollowText (feature clamped) contrast is designed to separate the two.
