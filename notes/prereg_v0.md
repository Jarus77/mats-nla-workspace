# Pre-registration v0: Do NLAs read non-workspace content, or re-solve from the text?

Draft 2026-10-05, **revision 2**. Revision 1 applied the lead/Fable design revision; revision 2 applies Fable's round-2 critique, approved by the user (§0, rows 11–22). Model: Qwen3.6-27B. Status: **draft for the user and Fable**. Freeze target: v1 on Oct 18, before any main-run data. Open choices are marked **[DECIDE: …]**. Companion notes: `pilot_deep_dive.md`, `lit_review.md`, `redteam.md`, `phase1_summary.md`, `wsbench_internals.md`.

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

**Co-headline.** NLA vs Oracle Lens under text-fixed steering (H3 + H4): does the NLA follow non-J features that OL does not, and is either reader explained by the text?

**Wording rule.** Claims say "outside the top-r J-space", never "outside the workspace".

## 2. The attribution frame (2×2 plus unsupported); used as the intro schematic

| | **Reader tracks the activation**: its output follows a feature moved with the text fixed | **Reader tracks the text**: its output follows a text change with the feature clamped |
|---|---|---|
| **Content in the J-space** (in-J features) | "Workspace reading". Expected for all three readers; this is the calibration. | "Re-solving / re-description of workspace-type content" |
| **Content outside the J-space** (non-J features) | **"Non-workspace reading". Neel's hypothesis for the NLA.** | "Re-solving / re-description of non-J-type content". E.g., counting characters in decoded text. |

Plus **no content / unsupported**: output that follows neither manipulation. Measured as the AUROC on the null contrast, plus the text-verifiable precision of free-form claims.

**Indices, per reader R and feature f (population level)**
- **FollowFeature(R, f, dose)** = AUROC with which R's paired log-odds separates +Δ from −Δ steered activations of the *same text*. Chance is 0.5.
- **FollowText(R, f)** = AUROC with which R separates two texts whose text-implied feature values differ, after **clamping** the probe projection of both to the same value.
- **Fraction of probe-recoverable information** = (AUROC_R − 0.5) / (AUROC_probe − 0.5), on the same contrast.
  - **Steering arm:** the probe direction is the steering direction, so AUROC_probe → 1 by construction and the fraction is ≤ 1. It reads as "how much of a guaranteed-present shift the reader conveys".
  - **Natural arm:** the probe is a *reference, not a ceiling*. A nonlinear reader could exceed a linear probe. A fraction > 1 is interpretable **only after H4 has ruled out re-solving**; otherwise it is attributed to text.
- **Role of the J-Lens.** In the non-J arm the J-Lens is the **leak detector**, not a competitor (§4.4). In the in-J arm it is a competitor.

## 3. Hypotheses (directional, with my prior)

| ID | Hypothesis | Directional prediction | Prior |
|---|---|---|---|
| **H1, calibration** | All three readers track in-J features under text-fixed steering at the doses used for non-J features. | FollowFeature ≥ 0.65 at the q99 dose for OL and NLA on ≥1 in-J feature. J-Lens ≥ 0.65 (it is a competitor in the in-J arm). | 0.75 |
| **H2, NLA reads non-J** (Neel's hypothesis) | In the read-layer steering arm, the NLA tracks natural non-J features at valid doses. | NLA FollowFeature 95% CI excludes 0.5 at q99 on ≥2 of 3 non-J features, **and the fraction of probe information is ≥ 0.25** (the hard bar). | **0.40.** Hong: the NLA must encode non-J content to reconstruct. Open: does the verbalizer condition on it? Viswanath's natural line count (3%) pulls the prior down. |
| **H3, co-headline: NLA vs OL** | Under text-fixed read-layer steering, the NLA tracks non-J features more than OL. OL is bound to the top-r J-space. | NLA − OL FollowFeature ≥ 0.05 on ≥2 non-J features (paired, item-clustered CI excludes 0). OL non-J ≤ 0.55 while OL in-J ≥ 0.65. If both read non-J, "workspace-bound" describes neither. | 0.40 |
| **H4, co-headline: text explains natural success** | On natural (unsteered) items, each prose reader's discrimination of non-J feature values is explained by the text. | Per reader: natural AUROC ≤ TextNLA(prompt) + 0.03, and (NLA) ≤ TextNLA(NLA's explanation) + 0.03. FollowText > FollowFeature. | 0.65 |
| **H5, re-solving control on intermediates (C)** | When an intermediate is planted by moving its J-removed or raw probe direction (prompt fixed, model answer flips), the prose readers follow the **text-implied** intermediate more often than the planted one. | P(planted) / (P(planted) + P(text-implied)) < 0.5 for OL and NLA. The J-Lens follows the planted value **only if** the probe direction has a J component. | 0.6 |
| **H6, dose monotonicity** | Where a reader tracks a feature, tracking rises from the q90 to the q99 dose (valid doses only). | Spearman ρ(dose, log-odds shift) > 0, Page's trend test p < 0.05. | 0.7 (conditional on any tracking) |

## 4. Operational definitions

### 4.1 J-space
- **Primary:** S_r = span of the top-r right singular vectors of J_L (the J-Lens Jacobian at the read layer L; SVD as in Viswanath).
  - **[DECIDE: r*]** Recommendation: report curves over r ∈ {64, 256, 1024, 2048}. Pick r* *a priori* as the smallest r at which a J-Lens built from S_r keeps ≥90% of the top-10 J-Lens readout agreement on the WorkspaceBench basic_readout bank.
  - Do **not** calibrate r* on any reader under test.
  - **G2 must pass at both r* and 4r*.** Curves over r are always shown.
- **Robustness:** Anthropic's per-activation sparse NNLS of J-Lens vectors (k = 25). Removal subtracts the fitted nonnegative combination.
- **Always report enrichment** = (energy in S_r) / (r/d), d = 5120. A random vector scores 1.0 (Viswanath: concept vectors 1.9×).
- **J artefact:** `neuronpedia/jacobian-lens` n1000 wikitext (the wsbench/pilot default). **[DECIDE/VERIFY: is `camilablank/workspace-lenses` J identical? Use one for both lens readout and J-space.]**

### 4.2 Feature criteria (gates)
Every gate is evaluated at the steering layer and at each read layer 40/42/43/44. **All features are gated, including the in-J features IJ1 and IJ2.**

| Gate | Non-J feature | In-J feature |
|---|---|---|
| G1 decodable | Held-out probe AUROC (binary) or Spearman (ordinal) ≥ 0.85 | Same |
| G2 J-removal | Probe trained on h, evaluated on h − P_{S}h: keeps ≥ 80% of (AUROC − 0.5), **and** a probe *retrained* on J-removed h reaches ≥ 0.80. **Must pass at r* and at 4r*.** | Retained ≤ 40% of (AUROC − 0.5) at r* **[DECIDE threshold]** |
| G3 J-Lens absence/presence | Label tokens (e.g. " forty", " past", " adjective") in the J-Lens top-20 at layers 36–48 at the read position at a rate ≤ the item-permuted null + 2pp | Label in the top-20 at ≥ 3× the null rate |
| G4 causal use (gate) | At the q99 dose, steering/ablating along the **J-removed** probe direction at L38 shifts the target behaviour beyond the equal-norm random non-J control (p < 0.01, item-paired) | The same, along the raw probe direction |
| Reported, **not gated** | Self-report under 3 elicitations: free text; logit 2AFC at the answer slot; few-shot | Same |

**G4 is a gate (decided).** Features that pass G1–G3 but fail G4 go to an appendix as **"decodable, unused"**, with natural-arm reads only.

**Steering direction for non-J features:** the **J-removed** probe direction u = normalise((I − P_{S_r*})w).
- P_S is the J-space at the layer where the vector is added (the read layer for the headline arm, L38 for the validation arm).
- This prevents injecting J content directly. It does **not** prevent downstream layers from rotating content into the J-space, hence the leak detector in §4.4.
- Robustness: u computed with 4r*.
- In-J features use the raw probe direction w.

### 4.3 Feature set

**Non-J candidates** (pick 2–3 that pass the gates; backups from the J-space appendix: capitalisation, coding language)

| ID | Feature | Items | Read position | Behavioural check (G4) |
|---|---|---|---|---|
| NJ1 | Characters since the last newline during **automatic linewrap** (Gurnee linebreaks); ordinal, 15–75 | 240 fixed-width passages × widths {30, 40, 50, 60, 70}, instruction "continue, preserving wrapping" | Token before each newline, plus the newline (≤4 positions/passage; cluster by passage) | Wrap column of the continuation shifts by ≥2 chars |
| NJ2 | **Verb tense** of the clause in automatic continuation (past vs present) | 240 minimal-pair stems | Stimulus tokens (J-space paper paired-question protocol, next-word condition) | Tense of the next generated verb flips at ≥20% over control |
| NJ3 | **Predicted POS of the next word** (adjective vs noun). **Ground truth = the model's predicted POS:** the POS-tagged probability mass of its next-token distribution (adjective mass vs noun mass, top-50 tokens tagged), not the word that actually follows. | 240 | Last prompt token | Adjective-vs-noun mass ratio shifts beyond control (log-ratio Δ, item-paired) |

**In-J calibration features** (2), **gated by G1–G4 like the non-J features**

| ID | Feature | Why |
|---|---|---|
| IJ1 | **Line count under explicit report** ("how many characters does the first line contain?") | The paper finds numbers enter the J-space at 20 positions under report, and the swap moves the answer 46 → 65. Same family as NJ1, so it controls for "readers can't talk about counts". |
| IJ2 | **A WorkspaceBench family with a known J-Lens token readout and regex ground truth:** **language identity** (multilingual_mt) **or the typo-correction target** (typo / typo_mt). | Stored numbers exist for both: typo_mt OL .83 / NLA .75 / J .17; multilingual_mt .62 / .44 / .06. Regex scoring needs no judge. **[DECIDE: language identity (binary, e.g. French vs Spanish) vs the typo target.]** Recommendation: language identity, a cleaner binary probe. |

### 4.4 Steering
**Two arms**

| Arm | Where the vector is added | Role |
|---|---|---|
| **R (headline): read-layer steering** | At the reader's input layer and position: L42 for all three readers (primary; D7); L40/L44 for OL as sensitivity. Layer convention = block-42 output (K0 passed, D5). Added to the captured vector before it is handed to the reader. No model forward pass after the edit. | **Reader-input test.** Does the reader's output follow a feature moved in its input, with everything else fixed? No downstream layers, so no leak path into the J-space. |
| **U (validation): upstream steering** | L38 at the read position(s), propagated through the model to 40–44. Also at later positions for behaviour (G4). | **On-distribution / behavioural validation.** The shift is produced by the model's own computation. Leak-prone, hence the leak detector. |

**Direction:** J-removed probe direction u (§4.2); raw w for in-J features.

**Doses**
- Let p be the natural projection u·h over the item bank at the steering layer, and m its median.
- Three doses Δ ∈ {q90 − m, q99 − m, 2(q99 − m)}, both signs (±Δ), plus 0.
- Equal-norm **random-direction controls** drawn from the same subspace class (random non-J for NJ, random J for IJ), plus isotropic random.

**Report per dose**
- ‖αu‖ / ‖h‖.
- Mean next-token KL on the clean continuation (arm U).
- **Probe-decodes-shift check:** the realised probe shift at each read layer (arm U). It must reach ≥ 50% of the injected Δ at the read layer, else the dose is reported as "did not propagate".

**Leak detector (pre-registered; arm U, and arm R as a sanity check)**
- J-Lens paired log-odds for the feature value at 40/42/44 on the ±Δ steered pair.
- **If FollowFeature(J-Lens) > 0.55 at any of 40–44, that dose is void for non-J claims.** It is reported as "leaked into the top-r J-space".
- J-Lens readouts are also checked at 4r*. **[DECIDE: 0.55 threshold.]**

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
- **Leak detector:** J-Lens FollowFeature ≤ 0.55 at 40–44 for any dose used in non-J claims.
- Probe-decodes-shift check (arm U).
- In-J calibration features.
- Explicit-report vs automatic condition (IJ1 vs NJ1).
- FVE as a one-sided guard; in-range evidence = K3 at matched doses.
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
| **K0** prep (Oct 7) | **Fix the DeltaNet kernels first** (pin `causal_conv1d` + fla). Then **functional tests (primary):** NLA FVE under each convention (match 0.761 vs the card's ~0.76); WorkspaceBench typo_mt (0.73 vs stored 0.75) and multilingual_mt (0.42 vs 0.44) at L42. Cosine vs `example_activations.parquet` is secondary (weak discriminator). Pipeline reproduces the pilot multihop100 numbers ±4. | Functional tests pick a convention → use it. **If cosine stays ≈ 0.974** with no clean separation, proceed with the FVE-matching convention and run a **sensitivity analysis under both conventions on the headline arm only**. Functional tests fail under both → stop NLA arms. |
| **K1** prep (Oct 12) | ≥2 non-J features pass G1–G4 at 40–44, and ≥1 in-J feature passes. | Only 1 non-J feature → run with 1 plus a backup (capitalisation/coding language). 0 → **pivot P1**. |
| **K2** sprint D2 | (a) **FVE guard** (one-sided): AR FVE within margin at q90/q99. (b) **Leak detector:** J-Lens FollowFeature ≤ 0.55 at 40–44 (arm U). (c) The probe shift reaches the read layer. | FVE fails at q99 → that feature runs at q90 only; fails at q90 → feature void. Leak at a dose → that dose is void for non-J claims in arm U (arm R unaffected). All void → **pivot P2**. |
| **K3** sprint D3 | **Dose-in-range evidence.** Calibration (H1): OL or NLA FollowFeature ≥ 0.65 on ≥1 in-J feature **at the same doses used for non-J features**. Random-direction null within 0.5 ± 0.04. | Fails → the readout is broken or the doses are too small. Switch to generation plus judged scoring (§6.2). Still fails → **pivot P2**. |
| **K4** sprint D7 | (C) yield: ≥60 items meet keep criteria (i) and (ii). | <60 → report (C) as descriptive, n stated; chain arithmetic at a ≥42 cell if one exists. |

- **P1 (no natural non-J feature at L42):** report the negative geometry result: "probe-decodable features at L42 are not outside the J-space under criterion X", with curves over r. Shift the headline to (C) plus the re-solving decomposition on WorkspaceBench (Design D: TextNLA, forced-choice rescoring).
- **P2 (doses invalid or readouts broken):** natural-only analysis (H4, panel 2) plus (C) plus the claim audit, framed as "how much of each reader's success is explained by text".

## 10. Decisive predictions by outcome

| Outcome pattern (valid doses) | Reading | Answer to Neel |
|---|---|---|
| NLA tracks non-J in arm R (CI > 0.5, fraction ≥ 0.25); OL does not; both track in-J at the same doses; TextNLA cannot explain natural success | The NLA reads content outside the top-r J-space that OL misses | **Yes**, at least for these features (the co-headline) |
| NLA tracks non-J **and** OL tracks non-J similarly | Non-J content is readable by any trained verbalizer; "workspace-bound" describes neither | "Not specific to NLAs" |
| NLA tracks non-J in arm R but not in arm U (leak-free doses) | The reader *can* decode the direction, but natural propagation does not produce a readable shift | Capability yes, natural relevance unclear |
| A positive in arm U only at doses where the leak detector fires | Content rotated into the J-space | Not evidence for non-J reading |
| No reader tracks non-J; all track in-J; NLA natural AUROC ≈ TextNLA | Verbal readers are workspace/text-bound; the NLA's extra content is re-description | **No.** The extra is text, not latent |
| No reader tracks in-J either | The method cannot see anything | Uninformative → K3 pivot |
| NLA FollowText > FollowFeature for non-J **and** in-J | The NLA is predominantly a text re-solver | No; consistent with the pilot |
| (C): readers follow planted ≥ text-implied | Readers read computed intermediates | Strengthens any H2 positive |
| (C): readers follow text-implied | Re-solving explains intermediate-naming | Consistent with the pilot, now on-distribution |

## 11. Figures

**Intro schematic.** The §2 2×2 (content location × reader mechanism), drawn as a diagram with no data.

**Figure 1 (headline): dose-response.**
- Two side-by-side panels: **one non-J feature** (e.g. NJ1, line characters under linewrap) and **one in-J feature** (e.g. IJ1, line count under explicit report).
- x = dose (0, q90, q99, 2·q99); y = FollowFeature AUROC (arm R) with 95% item-clustered CIs.
- Three lines per panel: NLA@42, OL@40/44, J-Lens@42.
  - In the non-J panel the J-Lens line is drawn as a **grey leak-detector trace**, not a competitor.
- Hollow markers = FVE-void or leaked doses. Thin dashed = arm U.
- Other features go to the supplement.

```
        non-J feature (NJ1)                      in-J feature (IJ1)
 AUROC                                    AUROC
 1.0 ┤                                    1.0 ┤             ╱── J-Lens
     │        NLA? ╱                          │        ╱───╱─── OL
 0.75┤        ╱───╱                       0.75┤   ╱───╱  ╱───── NLA
     │   ╱───╱     OL?                        │ ╱  ╱───╱
 0.5 ┼──●─────●─────●─────○  J-Lens (leak)  0.5 ┼●─────●─────●─────○
     └──0────q90───q99──2q99                  └0────q90───q99──2q99
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
| Oct 10–12 | Build item banks (NJ1–3, IJ1–2; IJ2 from multilingual_mt or typo). Train probes at 38/40/42/43/44. J SVD spectrum and the r* rule; G2 at r* and 4r*. Run gates G1–G4 on every feature, including the in-J ones → **K1**. |
| Oct 13–14 | Steering infrastructure for arms R and U. Natural spreads and doses. Leak detector, clamp validation (INLP fallback), FVE guard on 20 items. |
| Oct 15–16 | (C) bank and bridge probes; flip-rate pilot (n = 50). TextNLA prompts. Judge prompts. User labels 300 claims. |
| Oct 17–18 | Dry run of every arm at n = 10. Fable review. **Freeze prereg v1** (commit hash recorded, outside this repo's history if no commits are allowed). |

**Sprint, 2 weeks** **[DECIDE: dates; assumed Oct 19 – Nov 1]**

| Day | Work | Gate |
|---|---|---|
| D1 | (A) arm R (headline) on non-J and in-J features: all doses, three readers, teacher-forced scoring | |
| D2 | Arm U: L38 steering, behaviour, leak detector, probe-shift check, FVE guard; void assignment | **K2** |
| D3 | In-J calibration at matched doses (both arms); random-direction nulls | **K3** |
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
6. **Steering leak.** In arm U, layers 38→42 can rotate non-J content into the J-space. Handled by J-removed directions, the leak detector, and arm R as the headline. Residual risk: leakage into directions outside the top-r* J-space but inside the 4r* tail. The leak detector is also run at 4r*.
7. **Clamp leaves the feature decodable.** Handled by retrained-probe validation and the INLP subspace clamp. Residual risk: a nonlinear encoding a linear probe misses.
8. **OOD despite percentile doses.** Readers normalise input direction (OL unit-norm ×16000; NLA norm-matched add), so even small steers change the direction the reader sees. The FVE gate and KL reporting cover this partly.
9. **Judge validity** for the secondary metrics. Gemini over-credits on wsbench (precision 0.43 vs hand-labelled 0.33).

## 14. Open decisions (collected)
**Resolved in revision 2**
- G4 is a gate.
- NLA iter 400 primary, 300 robustness.
- Arm R is the headline, L38 (arm U) the validation.
- IJ2 comes from a WorkspaceBench family.
- The projection arm is in the appendix.

**Still open**
- r* rule details and the J artefact (neuronpedia vs workspace-lenses).
- G2 in-J threshold.
- IJ2: language identity vs typo target.
- Leak threshold 0.55.
- Clamp-validation threshold 0.55 and INLP k cap.
- FVE margin 0.05.
- Templates per reader (3).
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
