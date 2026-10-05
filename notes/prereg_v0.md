# Pre-registration v0: Do NLAs read non-workspace content, or re-solve from the text?

Draft 2026-10-05, revision 1, which applies the lead/Fable design revision (see §0). Model: Qwen3.6-27B. Status: **draft for the user and Fable**. Freeze target: v1 on Oct 18, before any main-run data. Open choices are marked **[DECIDE: …]**. Companion notes: `pilot_deep_dive.md`, `lit_review.md`, `redteam.md`, `phase1_summary.md`, `wsbench_internals.md`.

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

---

## 1. Research question

Neel: *"NLAs seem to capture more than J-lens/Oracle Lens. Is that really the non-workspace stuff?"*

**Operational question.** At matched depth (≈ layer 42 of 64) on natural text:
1. Does each reader's output **follow a feature when we move it in the activation with the text held fixed**?
2. Does it **follow the text when we move the text with the feature clamped**?
3. Does this differ between features that live **outside** the J-space and features that live **inside** it?

Readers: NLA (layer 42 only), Oracle Lens (OL; trained at 20–60 step 4), J-Lens.

**Where we stand relative to the pilot** (paragraph 1 of the write-up). The concurrent MATS pilot `LittleMeHere/nla-vs-lenses` found that on WorkspaceBench intermediates OL ≥ NLA ≫ J-Lens, and read the residual as prompt re-solving. **What's new here:** natural non-J features, causal steering with the text fixed, and on-distribution doses.

## 2. The attribution frame (2×2 plus unsupported)

| | **Reader tracks the activation**: its output follows a feature moved with the text fixed | **Reader tracks the text**: its output follows a text change with the feature clamped |
|---|---|---|
| **Content in the J-space** (in-J features) | "Workspace reading". Expected for all three readers; this is the calibration. | "Re-solving / re-description of workspace-type content" |
| **Content outside the J-space** (non-J features) | **"Non-workspace reading". Neel's hypothesis for the NLA.** | "Re-solving / re-description of non-J-type content". E.g., counting characters in decoded text. |

Plus **no content / unsupported**: output that follows neither manipulation. Measured as the AUROC on the null contrast, plus the text-verifiable precision of free-form claims.

**Indices, per reader R and feature f (population level)**
- **FollowFeature(R, f, dose)** = AUROC with which R's paired log-odds separates +Δ from −Δ steered activations of the *same text*. Chance is 0.5.
- **FollowText(R, f)** = AUROC with which R separates two texts whose text-implied feature values differ, after **clamping** the probe projection of both to the same value.
- **Fraction of probe-recoverable information** = (AUROC_R − 0.5) / (AUROC_probe − 0.5), on the same contrast.
  - The probe is the reference ceiling, not a hard bound. A nonlinear reader could exceed a linear probe, so values > 1 are allowed and reported.

## 3. Hypotheses (directional, with my prior)

| ID | Hypothesis | Directional prediction | Prior |
|---|---|---|---|
| **H1, calibration** | All three readers track in-J features under text-fixed steering. | FollowFeature ≥ 0.65 at the 99th-percentile dose for OL and NLA on ≥1 in-J feature. J-Lens ≥ 0.65 by construction. | 0.75 |
| **H2, Neel's hypothesis (NLA reads non-J)** | The NLA tracks natural non-J features under text-fixed steering at valid doses. | NLA FollowFeature 95% CI excludes 0.5 at the 99th dose on ≥2 of 3 non-J features. Fraction of probe information ≥ 0.25. NLA > OL by ≥ 0.05 AUROC. | **0.25** (Viswanath line count: NLA within ±3 of truth on 3%) |
| **H3, OL is workspace-bound** | OL tracks in-J but not non-J features. | OL non-J FollowFeature ≤ 0.55 at every valid dose, while OL in-J ≥ 0.65. | 0.55 |
| **H4, natural success is text** | On natural (unsteered) items, the NLA's discrimination of non-J feature values is explained by text. | NLA natural AUROC ≤ TextNLA(prompt) + 0.03, and ≤ TextNLA(NLA's explanation) + 0.03. FollowText(NLA) > FollowFeature(NLA). | 0.65 |
| **H5, re-solving control on intermediates (C)** | When an intermediate is planted by moving its probe direction (prompt fixed, model answer flips), the prose readers follow the **text-implied** intermediate more often than the planted one. | P(planted) / (P(planted) + P(text-implied)) < 0.5 for OL and NLA. The J-Lens follows the planted value **only if** the probe direction has a J component. | 0.6 |
| **H6, dose monotonicity** | Where a reader tracks a feature, tracking rises monotonically from the 90th to the 99th dose (FVE-valid doses only). | Spearman ρ(dose, log-odds shift) > 0, Page's trend test p < 0.05. | 0.7 (conditional on any tracking) |

## 4. Operational definitions

### 4.1 J-space
- **Primary:** S_r = span of the top-r right singular vectors of J_L (the J-Lens Jacobian at the read layer L; SVD as in Viswanath).
  - **[DECIDE: r*]** Recommendation: report curves over r ∈ {64, 256, 1024, 2048}. Pick r* *a priori* as the smallest r at which a J-Lens built from S_r keeps ≥90% of the top-10 J-Lens readout agreement on the WorkspaceBench basic_readout bank.
  - Do **not** calibrate r* on any reader under test.
- **Robustness:** Anthropic's per-activation sparse NNLS of J-Lens vectors (k = 25). Removal subtracts the fitted nonnegative combination.
- **Always report enrichment** = (energy in S_r) / (r/d), d = 5120. A random vector scores 1.0 (Viswanath: concept vectors 1.9×).
- **J artefact:** `neuronpedia/jacobian-lens` n1000 wikitext (the wsbench/pilot default). **[DECIDE/VERIFY: is `camilablank/workspace-lenses` J identical? Use one for both lens readout and J-space.]**

### 4.2 Feature criteria (gates)
Every gate is evaluated at the steering layer and at each read layer 40/42/43/44.

| Gate | Non-J feature | In-J feature |
|---|---|---|
| G1 decodable | Held-out probe AUROC (binary) or Spearman (ordinal) ≥ 0.85 | Same |
| G2 J-removal | Probe trained on h, evaluated on h − P_{S_r*}h: keeps ≥ 80% of (AUROC − 0.5), **and** a probe *retrained* on J-removed h reaches ≥ 0.80 | Retained ≤ 40% of (AUROC − 0.5) **[DECIDE threshold]** |
| G3 J-Lens absence/presence | Label tokens (e.g. " forty", " past", " adjective") in the J-Lens top-20 at layers 36–48 at the read position at a rate ≤ the item-permuted null + 2pp | Label in the top-20 at ≥ 3× the null rate |
| G4 causal use (steering validity) | At the 99th dose, steering along the probe direction shifts the target behaviour beyond the random-direction control (p < 0.01, item-paired) | Same |
| Reported, **not gated** | Self-report under 3 elicitations: free text; logit 2AFC at the answer slot; few-shot | Same |

**[DECIDE: is G4 a gate, or reported only?]** Recommendation: gate. A probe direction that does not move behaviour may not be the direction the model uses, and steering it would test nothing.

**Steering direction for non-J features:** the J-orthogonalised probe direction u = normalise((I − P_S)w), as in the J-space paper's appendix. A non-J steer then cannot add J content by construction. In-J features use the raw probe direction w.

### 4.3 Feature set

**Non-J candidates** (pick 2–3 that pass the gates; backups from the J-space appendix: capitalisation, coding language)

| ID | Feature | Items | Read position | Behavioural check (G4) |
|---|---|---|---|---|
| NJ1 | Characters since the last newline during **automatic linewrap** (Gurnee linebreaks); ordinal, 15–75 | 240 fixed-width passages × widths {30, 40, 50, 60, 70}, instruction "continue, preserving wrapping" | Token before each newline, plus the newline (≤4 positions/passage; cluster by passage) | Wrap column of the continuation shifts by ≥2 chars |
| NJ2 | **Verb tense** of the clause in automatic continuation (past vs present) | 240 minimal-pair stems | Stimulus tokens (J-space paper paired-question protocol, next-word condition) | Tense of the next generated verb flips at ≥20% over control |
| NJ3 | **Part of speech of the next word** (adjective vs noun) under a next-word instruction | 240 | Last prompt token | POS of the next word flips at ≥20% over control |

**In-J calibration features** (2) **[DECIDE]**

| ID | Feature | Why |
|---|---|---|
| IJ1 | **Line count under explicit report** ("how many characters does the first line contain?") | The paper finds numbers enter the J-space at 20 positions under report and the swap moves the answer 46 → 65. Same feature family as NJ1, so it controls for "readers can't talk about counts". |
| IJ2 | A **verbalizable concept, binary**: the bridge category in WorkspaceBench multihop pairs (e.g. metal vs gas element; month vs season). Read at the bank token. | J-Lens shows multihop bridges at L42–44 in 39/100. Overlaps with (C). |

Alternative IJ2: passage topic, animal vs vehicle. **[DECIDE]**

### 4.4 Steering
- **Where:** add α·u to the residual at layer L_s, at the read position(s). Also at subsequent positions for generation-based behaviour checks.
  - **[DECIDE: L_s = 38]** Recommendation: 38, upstream of every read layer. The realised projection shift is measured at 40/42/43/44 and used as the dose covariate.
  - Secondary arm: at-site steering at each read layer.
- **Doses:** let p be the natural projection u·h over the item bank at L_s, and m its median.
  - Three doses Δ ∈ {q90 − m, q99 − m, 2(q99 − m)}, with both signs (±Δ), plus 0.
  - Equal-norm **random-direction controls**, drawn from the same subspace as u (non-J random for NJ, J-random for IJ), plus isotropic random.
- **Report per dose:** ‖αu‖ / ‖h‖; mean next-token KL on the clean continuation; the realised probe shift at each read layer.
- **OOD check (pre-registered):** NLA reconstructor FVE on steered vs natural activations, same items.
  - A dose is **void for (A)** if mean FVE(steered) < mean FVE(natural) − **0.05** absolute, or < the 5th percentile of natural per-item FVE.
  - Same check with the OL AR (`agu18dec/olens_and_ar` AR) for OL.
  - **[VERIFY: an AR checkpoint for the ceselder NLA exists and loads; else use the OL AR as the shared OOD gauge.]**
  - **[DECIDE: margin 0.05?]**
- **Text-moved, feature-clamped (FollowText):** pair texts i and j whose feature values differ, e.g. a 40-char vs a 60-char line. Clamp each activation's projection to a common value c (the bank median): h' = h + (c − u·h)u, applied at L_s. Readers see the two texts' activations with matched feature projection.

### 4.5 (C) Re-solving control: planted intermediates
- **Items:** WorkspaceBench multihop (100) plus a constructed two-hop bank. Target 300 candidates, ≥120 kept.
  - Chain arithmetic is a secondary option, but no reader is above chance at L42 (pilot), so only if a ≥42 cell exists. **[DECIDE]**
- **Probe:** a multi-class bridge-identity probe at layer L_p ∈ {30, 34, 38}, trained on paraphrase-diverse prompts sharing a bridge.
- **Planting:** move the projection from the own bridge onto a partner bridge along the probe directions (difference of class means), at doses inside the natural spread. **No J-Lens token directions; no residual patches.**
- **Keep an item only if:**
  - (i) the model's answer flips to the partner's answer in ≥50% of planted runs;
  - (ii) the **prompt content stays decodable**: a subject-identity probe at L42 still reads the original subject (AUROC ≥ 0.9), and TextNLA-on-NLA-explanation recovers the original subject at the same rate as unplanted.
- **Score:** paired log-odds of planted vs text-implied bridge, in native format, for each reader. TextNLA(prompt) defines the pure re-solving prediction (always text-implied).
- **Split arm (optional) [DECIDE]:** plant using only the J component vs only the non-J component of the probe difference (cf. the J-space paper: 61% vs 28% flips). This gives the content-location axis for intermediates.

## 5. Readers, layers, matched comparisons

| Reader | Layers read | Primary cell | Notes |
|---|---|---|---|
| NLA `ceselder/qwen3.6-27b-nla-rl` | 42 only | 42 | `av_rl_adapters/iter_000400` (pilot, wsbench). **[DECIDE: 300 per the model card?]** Run the off-by-one check (§9, K0) before anything else. |
| Oracle Lens s3d-rl600 (open model-repo copy) | 40, 44 (on-contract); 42 (off-contract, reported) | max(40, 44) and mean(40, 44) | `{layer}` in the prompt; capture with the adapter disabled. |
| J-Lens (neuronpedia n1000) | 40, 42, 44 (and 20..60 step 4 for presence) | 42 | Label-token direction scores. |
| Linear probe | 40, 42, 43, 44 | The read layer of the compared reader | The ceiling. |
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
- **J-Lens:** max cosine score over each value's label-token variants, s(a) − s(b).
- **Probe:** the projection.
- **TextNLA:** the same NLA-register template, conditioned on the text.
- **[DECIDE: 3 templates per reader, averaging LO; report per-template spread.]**

**AUROC is the common currency.**
- FollowFeature: AUROC of LO between the +Δ and −Δ steered versions of the same items.
- FollowText: AUROC between the clamped texts.
- Natural: AUROC between items with high vs low natural values.
- **Matched nulls:**
  - (i) LO on 0-dose activations: the reader's prior/bias, which subtracts out of the paired contrast;
  - (ii) equal-norm random-direction steering (same subspace class): must give AUROC ≈ 0.5;
  - (iii) item-permuted candidates: score against other items' values (adopted from the pilot).

**Statistics**
- Item-clustered bootstrap with 2,000 resamples, clustering by passage, plus a per-feature mixed model `LO ~ dose × reader + (1 | item)`.
- Holm correction across the 3 non-J features within each hypothesis.

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
- Clamp-only (no text change), to check the clamp itself is inert.
- In-J calibration features.
- Explicit-report vs automatic condition (IJ1 vs NJ1).
- FVE-based OOD gate.
- TextNLA (prompt) and TextNLA (NLA explanation).
- Layer-matched probe ceilings at 40/42/43/44.
- Pilot replication on multihop100 as a pipeline check (expect J 39 / OL 80 / NLA 75, ±4).

## 8. Sample sizes and power
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
- **Clustering:** NJ1 has up to 4 positions per passage. With ICC ≈ 0.5 the design effect is 1 + (4 − 1)·0.5 = 2.5. So 240 passages give an effective n ≈ 380 positions ≈ 190 per class after splitting.
- **Plan: 240 items per feature**, which detects AUROC 0.60 with power ≥ 0.9. We will *not* claim effects of AUROC ≤ 0.55; that is the pre-declared "no meaningful tracking" region.
- **NLA vs OL difference** (paired, same items): about 0.05 AUROC detectable at n ≈ 240 if the between-reader correlation ≥ 0.5 **[compute exactly on prep data]**.
- **(C):** 120 kept items. P(planted) CI half-width ≈ ±0.09 at p = 0.5. McNemar planted vs text-implied with discordant rates 0.20 / 0.05 needs n = 85.
- **Compute:** about 5 features × 240 items × 7 doses × (3 readers + TextNLA) × 2 candidates × 3 templates ≈ 200k short teacher-forced passes, plus about 15k generations (k = 5) and the AR FVE passes. A few H100-days on Modal; cheap.

## 9. Kill criteria and pivots

| When | Check | Kill / pivot |
|---|---|---|
| **K0** prep (Oct 7) | NLA layer convention: our capture vs the NLA repo's `data/example_activations.parquet` (cosine for block-41 vs block-42 output), plus NLA FVE on both. Pipeline reproduces the pilot multihop100 numbers ±4. | Mismatch unresolved → **stop all NLA arms** until fixed. |
| **K1** prep (Oct 12) | ≥2 non-J features pass G1–G4 at 40–44, and ≥1 in-J feature passes. | Only 1 non-J feature → run with 1 plus a backup (capitalisation/coding language). 0 → **pivot P1**. |
| **K2** sprint D2 | OOD gate: the NLA AR FVE stays within margin at the 90th and 99th doses for the passing features. | The 99th fails for a feature → that feature runs at the 90th only. The 90th fails → feature void for (A). All void → **pivot P2**. |
| **K3** sprint D3 | Calibration (H1): OL or NLA FollowFeature ≥ 0.65 at the 99th dose on ≥1 in-J feature. Random-direction null within 0.5 ± 0.04. | Fails → the readout is broken. Switch to generation plus judged scoring (§6.2). Still fails → **pivot P2**. |
| **K4** sprint D7 | (C) yield: ≥60 items meet keep criteria (i) and (ii). | <60 → report (C) as descriptive, n stated; chain arithmetic at a ≥42 cell if one exists. |

- **P1 (no natural non-J feature at L42):** report the negative geometry result: "probe-decodable features at L42 are not outside the J-space under criterion X", with curves over r. Shift the headline to (C) plus the re-solving decomposition on WorkspaceBench (Design D: TextNLA, forced-choice rescoring).
- **P2 (doses invalid or readouts broken):** natural-only analysis (H4, panel 2) plus (C) plus the claim audit, framed as "how much of each reader's success is explained by text".

## 10. Decisive predictions by outcome

| Outcome pattern (valid doses) | Reading | Answer to Neel |
|---|---|---|
| NLA tracks non-J (CI > 0.5, fraction ≥ 0.25); OL does not; all track in-J; TextNLA cannot explain it | The NLA reads non-workspace content OL misses | **Yes**, at least for these features |
| NLA tracks non-J **and** OL tracks non-J similarly | Non-J content is readable by any trained verbalizer; OL is not workspace-bound | "Not specific to NLAs" |
| No reader tracks non-J; all track in-J; NLA natural AUROC ≈ TextNLA | Verbal readers are workspace/text-bound; the NLA's extra content is re-description | **No.** The extra is text, not latent |
| No reader tracks in-J either | The method cannot see anything | Uninformative → K3 pivot |
| NLA FollowText > FollowFeature for non-J **and** in-J | The NLA is predominantly a text re-solver | No; consistent with the pilot |
| (C): readers follow planted ≥ text-implied | Readers read computed intermediates | Strengthens any H2 positive |
| (C): readers follow text-implied | Re-solving explains intermediate-naming | Consistent with the pilot, now on-distribution |

## 11. Figures

**Figure 1 (headline).** One panel per reader (NLA@42; OL@40/44; J-Lens@42).
- x = steering dose (0, q90, q99, 2·q99); y = FollowFeature AUROC with 95% clustered CIs.
- Solid lines = non-J features; dashed = in-J features.
- A grey band for the probe ceiling per feature at the matched layer.
- Hollow markers = FVE-void doses.

```
 AUROC
 1.0 ┤ ─ ─ probe ceiling ─ ─ ─ ─ ─
     │               ╱─── in-J (IJ1, IJ2)
 0.75┤          ╱───╱
     │     ╱───╱      ┌── non-J (NJ1–3)?
 0.5 ┼────●─────●─────●─────○      ← random-direction null
     └────0────q90───q99──2q99  dose
        NLA@42        OL@40/44       J-Lens@42
```

**Figure 2 (what re-solving explains).** Grouped bars per non-J feature on natural items: NLA, TextNLA | prompt, TextNLA | NLA explanation, OL, J-Lens, probe. Inset: FollowText vs FollowFeature per reader (the 2×2 mechanism axis).

**Figure 3 (the 2×2).** A scatter of FollowFeature (x) vs FollowText (y) per reader × feature. Colour = content location (in-J / non-J). The quadrants are labelled with the §2 cells.

**Figure 4 (C).** P(planted) vs P(text-implied) per reader, with the TextNLA prediction marked; optional J-only / non-J-only planting.

**Supplementary.**
- Probe AUROC at 40/42/43/44 and its retention under J-removal across r, with enrichment.
- Dose diagnostics: ‖αu‖/‖h‖, next-token KL, AR FVE.
- Self-report rates per feature (reported, not gated).
- Text-verifiable precision per reader.

## 12. Timeline

**Prep, Oct 5–18**

GPU work is allowed in prep. **[DECIDE: confirm MATS rules on pre-sprint work.]**

| Dates | Work |
|---|---|
| Oct 5–6 | Modal capture pipeline (capture-once, dual 42/43). Pin kernels. Reproduce pilot multihop100 (K0 part 2). |
| Oct 7 | **K0:** NLA layer convention via `example_activations.parquet` plus FVE; locate the AR checkpoint. |
| Oct 8–9 | Paired log-odds scorer for every reader. Mine and freeze templates on a calibration split. Sanity-check on basic_readout (positive control). |
| Oct 10–12 | Build item banks (NJ1–3, IJ1–2). Train probes at 38/40/42/43/44. J_42 SVD spectrum and the r* rule. Run gates G1–G4 → **K1**. |
| Oct 13–14 | Steering infrastructure. Natural spreads and doses. First AR FVE OOD check on 20 items. |
| Oct 15–16 | (C) bank and bridge probes; flip-rate pilot (n = 50). TextNLA prompts. Judge prompts. User labels 300 claims. |
| Oct 17–18 | Dry run of every arm at n = 10. Fable review. **Freeze prereg v1** (commit hash recorded, outside this repo's history if no commits are allowed). |

**Sprint, 2 weeks** **[DECIDE: dates; assumed Oct 19 – Nov 1]**

| Day | Work | Gate |
|---|---|---|
| D1 | (A) full run on non-J features: all doses, three readers, teacher-forced scoring | |
| D2 | AR FVE on every dose; void assignment | **K2** |
| D3 | (A) in-J calibration features; random-direction nulls | **K3** |
| D4 | FollowText (clamp) arm; TextNLA (prompt / NLA explanation); natural AUROCs | |
| D5 | First-pass Figures 1–3; write a one-page interim for Fable/Neel | |
| D6 | (C) planting at full n; behaviour flips; keep criteria | |
| D7 | (C) reader scoring; checkpoint, re-plan | **K4** |
| D8 | Free-form k = 5 generations; judge runs; judge validation κ | |
| D9 | Robustness: r sweep, NNLS J definition, OL at 42, at-site steering, NLA iter 300 vs 400 | |
| D10 | Robustness: templates, a second TextNLA model, layer 43 convention | |
| D11 | Optional: 20-item split-injection replication (only if D1–D10 are on track) | |
| D12 | Final figures, statistics, error bars | |
| D13 | Write-up: Neel-style summary first, then details. Pilot cited in paragraph 1 | |
| D14 | Buffer; red-team the write-up with Fable | |

## 13. Risks
1. **Readers fail for uninteresting reasons.**
   - They were never trained to verbalize character counts or POS, and the paired templates may sit outside their output distribution.
   - A non-J null is then uninterpretable.
   - Mitigations: in-J calibration in the *same* family (IJ1 vs NJ1), templates mined from each reader's own outputs, judged free-form scoring as a cross-check, the K3 gate.
2. **Feature validity.** Natural features may not pass G2/G4 on Qwen3.6-27B at L42, or the probe direction may not be the causal one.
   - Steering may need doses beyond the FVE-valid range to move behaviour.
   - G4 plus the FVE gate may leave few valid cells.
3. **"J-space" is a modelling choice.** A heavy-tailed spectrum with no elbow: on Llama the conclusions flipped between r = 200 and r = 2000, and a random vector puts 2.4% inside r = 200.
   - What counts as "non-J" depends on r and on the J artefact.
   - Mitigations: curves over r, an a-priori r* rule, NNLS robustness, enrichment reporting.
4. **NLA layer off-by-one** (Viswanath's Anthropic NLA uses the `hidden_states[L]` convention). If it is wrong, every NLA number is off-layer → K0.
5. **Novelty and overlap.**
   - Viswanath's natural line count is already negative for the NLA on Llama (3%).
   - His unposted follow-up does reader-side J/non-J reads on natural safety activations.
   - The pilot owns split-injection.
   - Our differentiators: three readers including OL, Qwen, text-fixed steering at on-distribution doses with an FVE gate, the FollowText clamp, TextNLA re-solving ceilings.
   - **[DECIDE: contact the pilot author?]**
6. **OOD despite percentile doses.** Readers normalise input direction (OL unit-norm ×16000; NLA norm-matched add), so even small steers change the direction the reader sees. The FVE gate and KL reporting cover this partly.
7. **Judge validity** for the secondary metrics. Gemini over-credits on wsbench (precision 0.43 vs hand-labelled 0.33).

## 14. Open decisions (collected)
- r* rule and the J artefact.
- G2 in-J threshold.
- Whether G4 is a gate.
- IJ2 choice.
- L_s = 38.
- FVE margin 0.05.
- NLA iter 300 vs 400.
- Templates per reader (3).
- Fireworks judge model.
- TextNLA robustness model.
- (C) chain-arithmetic fallback and the J/non-J split planting arm.
- Sprint dates and pre-sprint rules.
- Contacting the pilot author.
