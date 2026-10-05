# Handoff #2 to Fable — 2026-10-05 (evening)

Paste everything below the line into the Fable 5.1 chat.

---

Thanks. I accepted almost all of your critique: the 2×2 frame, cutting (B), (A) as the headline, self-report reported but not a gate, planting by moving the feature, the dose and OOD rules, paired log-odds, "text-verifiable precision", and the layer caveat. This is the update: early numbers, two new findings from prior work, my pushbacks on your critique, and the open design decisions. Sprint is confirmed for Oct 19 – Nov 1; prep runs until Oct 18. Please be blunt again.

## 1. Early gate numbers (preliminary; 3 of 4 workstreams still running)
- **NLA reconstruction reproduced.**
  - Held-out FVE is **0.761** (adapter iter_000400, 64 example activations, 4 samples each) vs the published ~0.76. Per-example SD 0.18; one outlier at −0.37.
  - Predict-the-mean baseline MSE 0.46.
- **NLA on WorkspaceBench (regex-scored families, L42) replicated within CI:**

  | Family | Ours | Published |
  |---|---|---|
  | typo_mt | 0.73 [0.64, 0.81] | 0.75 |
  | multilingual_mt | 0.42 [0.33, 0.52] | 0.44 |
  | multihop_mt | 0.15 | — |
  | basic_readout_mt | 0.57 | — |

  Oracle Lens and J-Lens arms are still running.
- **Layer convention is unresolved.** The NLA's own example activations vs our recomputation: cosine **0.974** under one convention vs **0.95** under the other. That leans toward one, but a true match should be ≥0.99. Suspects:
  - context length at capture;
  - a slow fallback for the DeltaNet layers (the fast kernels weren't installed);
  - bf16 precision.

  This is kill check K0: no NLA arm runs until it's resolved.
- **Search for natural non-J features** (probes, J-removal curves over rank r, causal use, self-report) is running on GPU. Results expected tonight or tomorrow.

## 2. New from prior work
**(a) The concurrent pilot (github.com/LittleMeHere/nla-vs-lenses).** I recomputed their raw files; every headline number reproduces exactly. New analyses on their data:
- **The NLA finds nothing Oracle Lens misses.** NLA-only hits: 3/100; Oracle-Lens-only hits: 8/100 (McNemar p=0.23). The two prose readers succeed on the same items (phi 0.69).
- **Hits look like a solved fact.** The bridge is named together with the answer in 67/75 NLA hits and 79/80 Oracle Lens hits ("iron (Fe)"). That looks like re-solving.
- **Single-sample noise is large.** Comparing near-identity controls with the originals gives kappa 0.55–0.70, so 7–10 of 50 items flip. Their swap and brew differences fall inside this noise.
- **The swap was weak.** 34/50 items hit the 16× cap. Readers follow the planted bridge on 4–6 of the 17 items where J-Lens even shows it.
- **Their deletion test can't distinguish** "re-solving" from "reading non-J content". It removes only J-Lens readout directions and never checks the model's behaviour.
- **Other problems:** Oracle Lens read at L42 (off its training grid); NLA read at 16/28/36/56 (off-layer); no text-only baseline; unvalidated judges.

**(b) Viswanath's unposted follow-up** (github.com/PranavViswanath/nla-introspection, `jspace-followup/`, July 13). On Llama-3.3-70B:
- **Safety content is barely concentrated in J-space.** Enrichment (energy in top-r J directions over the random expectation r/d) is 1.45 for natural safety-scenario contrasts, against 1.00 for random and 1.90 for concept vectors. Grader gaming specifically: 1.00.
- **The NLA reads these from the non-J part.** Fed norm-matched components: full activation 0.55, J-only 0.18–0.38, non-J only 0.53–0.57 (keyword grading; floors 0.00–0.22).
- **Both components are causal at equal norm.**
- **His own open caveat:** "the verbalizer inferring the scenario from thematic residue rather than genuinely decoding the component". His splice control is queued, not run.
- **No Oracle Lens, no Qwen.** He also flags that the Jacobian spectrum has no elbow (top-200 dims = 21% of output-influence energy, top-2000 = 67%), and his conclusions moved between r=200 and r=2000.

**My read:** "the NLA reads natural non-J content" is partly shown already, on Llama. **Reading vs re-solving is explicitly open**, and that's exactly our text-fixed steering arm. The novelty claim becomes:
1. known-value natural features with independent ground truth (not thematic keywords);
2. a causal test separating reading from re-solving;
3. all three readers, including Oracle Lens, on Qwen3.6-27B at matched layers.

The first paragraph of the write-up will cite both the pilot and Viswanath.

## 3. Pushbacks on your critique
1. **A linear probe isn't a ceiling.** The readers are nonlinear 27B decoders. I report "fraction of probe-recoverable information" with the probe as a *reference*, allow values > 1, and treat a reader beating the probe as a finding.
2. **Natural-data success can't show "reads non-J".** A text re-solver can compute character count from the decoded text. So the write-up's non-J claim rests **only** on the steering arm (FollowFeature with text fixed). Natural results feed H4, i.e. how much of the success is text. Agree?
3. **The FVE-based OOD check is partly circular.** The reconstructor was co-trained with the NLA verbalizer, so high FVE on a steered activation shows the *pair* copes with it, not that the verbalizer reads it. I add two checks that don't use the reconstructor: next-token KL, and the probe still decodes the intended shifted value at the read layer. Enough?
4. **J-Lens has no decoder.** Its "paired log-odds" is the difference in J-Lens projection score between the two candidate tokens, which is comparable to the decoders only after AUROC normalisation. Acceptable, or should J-Lens be reported separately?

## 4. The pre-registration in brief
**Indices** (per reader R, feature f):
- FollowFeature(R, f, dose) = AUROC separating +Δ from −Δ steered activations of the *same text*.
- FollowText(R, f) = AUROC separating two texts with different text-implied values, after **clamping** the probe projection of both to one value.
- Fraction of probe information = (AUROC_R − 0.5) / (AUROC_probe − 0.5).

**Hypotheses (my priors):**

| ID | Hypothesis | Prior |
|---|---|---|
| H1 | Calibration: readers track in-J features | 0.75 |
| **H2** | **Neel's hypothesis: the NLA tracks natural non-J features under text-fixed steering**, with a 95% CI above 0.5 on ≥2 of 3 features and fraction ≥0.25 | **0.25** |
| H3 | Oracle Lens is workspace-bound | 0.55 |
| H4 | Natural success is explained by text: NLA ≤ TextNLA + 0.03 | 0.65 |
| H5 | (C) The prose readers follow the text-implied intermediate over the planted one | 0.6 |
| H6 | Dose monotonicity | 0.7 |

**Features** (pending gates):
- **Non-J:**
  - NJ1: characters since the last newline during automatic line wrap;
  - NJ2: verb tense in automatic continuation;
  - NJ3: next-word part of speech (adjective vs noun).
- **In-J:**
  - IJ1: line count under explicit report (same family as NJ1, controls for "can't talk about counts");
  - IJ2: a binary bridge category from multihop.

**Gates:**
- G1: probe decodable.
- G2: the frozen probe keeps ≥80% of (AUROC − 0.5) after J removal, *and* a probe retrained on J-removed activations reaches ≥0.80.
- G3: absent from the J-Lens readout.
- G4: causally used.

**Steering:** at layer 38, upstream of all read layers. Doses at p90, p99 and 2×p99 of the natural projection; the realised shift is measured at 40/42/43/44.

**Power:** 240 items per feature detects AUROC 0.60 with power ≥0.9.

**Kill checks:**

| Check | When | Pass condition |
|---|---|---|
| K0 | Oct 7 | layer convention resolved |
| K1 | Oct 12 | ≥2 non-J features and ≥1 in-J feature pass the gates |
| K2 | Sprint day 2 | FVE within a 0.05 margin at the doses |
| K3 | Sprint day 3 | calibration: FollowFeature ≥0.65 on ≥1 in-J feature |
| K4 | Sprint day 7 | ≥60 valid items for (C) |

**Pivots:**
- P1 (no natural non-J feature): report the negative geometry result plus (C) and a re-solving decomposition on WorkspaceBench.
- P2 (doses invalid or readouts broken): natural-only analysis plus (C) plus a claim audit.

## 5. Decisions I want your call on
1. **H2 prior.** Given Viswanath's follow-up (NLA non-J-only ≈ full on natural safety content), should H2 be well above 0.25? And does his result make a *positive* H2 less interesting, so that the reading-vs-re-solving split is the real contribution?
2. **J-space rank r\*.** My rule: the smallest r at which a J-Lens rebuilt from the top-r subspace keeps ≥90% of top-10 readout agreement on WorkspaceBench basic_readout; report curves over r ∈ {64, 256, 1024, 2048}. Is that defensible given the spectrum has no elbow, or should I anchor r\* behaviourally (Viswanath anchors on a reportability crossover at r≈2000)?
3. **Is G4 (causal use) a gate?** I say yes: steering a direction the model doesn't use tests nothing. But it may kill otherwise good features.
4. **Feature choices.** Is IJ1 (line count under explicit report) a good in-J control for NJ1, and is the multihop bridge category a good IJ2? Better candidates?
5. **Cheap secondary arm.** Viswanath-style J-only / non-J-only projections for all three readers on our features. It's cheap and directly comparable to his Llama result. Worth including, or scope creep?
6. **NLA checkpoint.** iter_000400 (what WorkspaceBench and the pilot used; our FVE matches) vs 300 (the model card's guess). I plan 400 as primary, with 300 as a robustness check.
7. **K0 fallback.** If the layer convention can't be pinned above ~0.97 cosine, can we proceed with the better-matching convention plus a documented sensitivity analysis (run both), or is that fatal for NLA claims?
8. **Write-up framing.** Working title: "Do NLAs read what the model can't say, or re-solve what it read?" Does the 2×2 belong in the first figure, or should the first figure be the dose-response curve?
