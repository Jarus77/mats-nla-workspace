# Research summary: do NLAs read what the model can't say?

Status as of 12 October 2026. Prep and pilot phase complete (5–12 October); the pre-registered confirmatory run is 19 October – 1 November (prereg revision 4.2, `notes/prereg_v0.md`). Every pilot number below comes from a declared pilot split that is excluded from the main analysis. The dated decision log is `notes/decisions.md` (D1–D33); per-experiment detail is in `experiments/NN_*/RESULTS.md`.

---

## 1. Research questions

**The question we were given (Neel Nanda, MATS 12.0):** *What can natural-language autoencoders (NLAs) capture that Oracle Lens and J-Lens don't? Is it really the stuff not in the workspace?*

The "workspace" is the J-Lens's picture of the residual stream: the set of directions that map, through the model's own unembedding Jacobian, onto tokens the model is poised to say. The hypothesis behind the question is that an NLA, which is trained to verbalise an activation and reconstruct it, might read content that lives outside that verbal workspace, while the J-Lens (a linear map to vocabulary) and Oracle Lens (the model itself plus a LoRA, reading an injected activation) cannot.

We split this into five answerable questions, all on Qwen3.6-27B at layer 42 (block-42 output), the layer the NLA was trained on:

| ID | Question | Where answered |
|---|---|---|
| Q1 | What counts as "not in the workspace", operationally, and can we certify a natural feature as such? | Part A, exps 02–04 |
| Q2 | When a certified non-workspace feature is moved inside the activation with the text held fixed, does any reader follow it? | Part B, exps 07–09 |
| Q3 | Do the readers read workspace features at the same steering strength (so that a null on Q2 is interpretable)? | Part B, exp 08 (K3) |
| Q4 | On the published benchmark (WorkspaceBench), is the readers' success evidence of reading the activation, or of re-solving the prompt text? | Part D, exps 01, 06 |
| Q5 | What *do* the readers read that the model represents, and in what form do they report it? | Part C, exps 05, 10 |

A sixth question arrived externally on 12 October (M. Kenney, nla-consistency): a linear probe reads arithmetic operands from the "=" position that the NLA under-reports. Is operand identity non-workspace content? (Exp 10.)

---

## 2. Methods

### 2.1 Subject model, readers, site

- **Model:** Qwen3.6-27B, 64 layers, d = 5,120, bf16. Read site L42 = output of decoder block 42 = HF `hidden_states[43]`, last token. Established as the NLA's training convention in exp 00 (K0): block-42 output reconstructs with cosine 0.9992 against 0.974 for the neighbouring blocks, 64/64 examples. FVE 0.761 reproduced.
- **NLA:** `ceselder/qwen3.6-27b-nla-rl`, adapter iter_000400 (the WorkspaceBench default; "RL step 400" in Kenney's terminology). Injects ‖h_marker‖·unit(v) at verbaliser block 1; scored by paired log-odds of frozen template continuations after `<explanation>\n`, or by free generation for judge-based checks.
- **Oracle Lens (OL):** Qwen + LoRA `olens_s3d_rl600`; the slot embedding is replaced by 16000·unit(h); outputs `- ` bullet continuations. Read at 42 (off its training grid of 20–60 step 4, but no worse than 40/44), with 40/44 as sensitivity.
- **J-Lens:** neuronpedia n1000 (primary) and camilablank n25 (cross-check); standard readout softmax(W_U·norm(J·h)) with gain 1+w. The two lenses differ substantially (vector cosine 0.79, top-25 overlap 19%), which is why every non-J label is checked under both.
- **J++ Lens** (Ayonrinde & Lindsey; `koayon/jpp-lenses`): a third basis, used in exp 10 only.
- **Judges** for free-text checks: Gemini 3.8 Flash (OpenRouter) and kimi-k3 (Fireworks); κ 0.72–0.77 judge vs judge on claim precision.

### 2.2 The J-space

J-space at rank r = the top-r right singular vectors of M = W_U·diag(1+w)·J_42. r\* = 384 is the smallest rank whose rebuilt J-Lens keeps ≥ 90% top-10 agreement with the full lens on WorkspaceBench basic_readout; 4r\* = 1,536 is the conservative rank used for all steering. There is no spectral elbow, so every claim is stated as "outside the top-r J-space" at both ranks.

### 2.3 Certifying "not in the workspace": the four-gate test (D9)

The standard test in the literature, "train a probe, project out the J-space, retrain, see if accuracy survives", turned out to be uninformative (§4.1). We replaced it with four gates, each with a matched-rank random null:

- **G1, decodable.** Linear probe (ridge for counts, multinomial logistic for classes; standardised h; L-BFGS; three document-grouped 80/20 splits; cluster-bootstrap CI). Must beat chance and a Hewitt–Liang control task (random label per token type), so the probe cannot pass by memorising token identity.
- **G2, geometrically outside J.** Enrichment of the feature's diff-in-means direction d (covariance direction for counts): ‖Q_rᵀd‖² ÷ (r/d). Random = 1.0; a generic centred activation scores 2.2 at r\* and 1.5 at 4r\*. Threshold ≤ 1.5 at both ranks, i.e. "less J-loaded than an ordinary activation direction". Shuffled-label directions give the floor. Checked under both lenses.
- **G3, not readable by the J-Lens, in the form the lens emits.** Score whether the tokens the lens actually emits track the feature's value: digits and number words for counts, label words or the part of speech of the top-10 tokens for categories, newline mass for col. Statistic: AUROC against a 200-permutation null. This gate caught predicted POS, which was unreadable on label words but readable at 0.66 through the token class of the lens's own next-word guesses.
- **G4, causally used.** Steer along the J-removed direction (feature direction with its top-4r\* J-component projected out), centred construction at a matched dose, and measure the model's own behaviour (newline log-odds for col, next-token tense readout for tense). Must exceed the maximum over random directions of the same norm, and must not leak: steering at L38 leaves the J-Lens readout at 40/42/44 inside its null band.

Labels: all four pass → **NON-J**; fail G2 or G3 with a readable lens → **IN-J**; pass G2 but marginally lens-readable → **DIFFUSE-BUT-J-READABLE**; pass G1–G3, fail G4 → **decodable-unused**.

### 2.4 Text-fixed steering ("FollowFeature")

The decisive measurement for Q2 and Q3. Keep the prompt text identical; move the feature inside the activation; ask whether each reader's output follows.

- **Construction:** centred steering h′ = h − (v·h)v ± Δv, equal norm on both sides. Additive steering h ± Δv produced an apparent OL anti-tracking that was a dilution artefact when h·v ≠ 0 (exp 03, D11); centred steering collapses it to chance.
- **Directions:** the J-removed (4r\*) direction for the non-J test; raw class-mean directions for the in-J calibrations.
- **Doses:** q90 − median of the natural projection (primary), q99 (secondary), and matched relative norms r11 = 0.11‖h‖ and r22 for cross-feature comparisons. The FVE guard (paired, k = 4, margin 0.05) voids doses the NLA cannot reconstruct.
- **Readout statistic:** paired log-odds shift ē = mean over items of (LO(+Δ) − LO(−Δ))/2 on frozen template pairs, plus paired win rate and cross-item AUROC.
- **Null bands:** the same statistic over K random directions at the same per-item norm, drawn isotropically and from the J-complement (K = 20 in the pilot, K = 100 in the main run). Results are reported as z against both bands; the band, not n, limits power.
- **Leak detector (two-sided):** a dose is void if the J-Lens's label set or a content set moves outside its K = 100 J-complement band. This certifies the dose, not the feature.
- **Behavioural validation:** the same edit must move the model's own behaviour (newline log-odds, wrap length in greedy generation, next-verb tense).

### 2.5 Pre-registration

Four primary cells, Holm-corrected, on fresh passage-disjoint items (240 col, 160 tense, 120 language): P1 NLA on J-removed col; P2 OL's next-token continuation channel on J-removed col; P3 OL on J-removed tense; P4 NLA on J-removed tense. Claim criterion: z_min ≥ +3 against both K = 100 bands, real effect above all 100 nulls, sign agreement within the declared secondary family, leak- and FVE-valid. Two positive controls (language, tense raw at r11), five pre-registered OL negatives (equivalence tests, TOST), a template-free judge check (T1, 6,720 free readouts), and declared secondaries S1–S10. Frozen inputs are recorded by sha256 in a manifest at the 18 October freeze. Four fixed-wording outcome branches (§5).

### 2.6 Infrastructure

Modal (app "rfm"; volumes rfm-weights, rfm-results); A100-80GB / H100 / H200; deployed functions spawned server-side with per-stage checkpoints so jobs survive the client. Library in `src/mats/` (readers, steering, J-space tools, metrics; `pytest tests/`). Prep-phase compute ≈ $330–390 plus ≈ $12 of judge calls.

---

## 3. Experiments

| Exp | Purpose | Scale | Status |
|---|---|---|---|
| 00 NLA gate | K0: layer convention; FVE | 64 activations | Established |
| 01 WorkspaceBench replication | Reproduce 12 published numbers; protocol audit | 4 families × 3 readers, 11 layers | Established |
| 02 Non-J feature gates | 8 features through G1–G4 at L42 (L40/44 sensitivity) | 4,000 rows/class; 40k regression rows | Established |
| 03 Pipeline smoke | End-to-end plumbing on a toy in-J feature (French vs German) | 40 pairs | Plumbing; produced the dilution finding |
| 04 Backup features | 9 further candidates through the same gate | ~20 H100-h | Established |
| 05 Planted intermediates | Bridge-entity swap at L38 as in-J calibration; NLA free explanations | 614 candidates, 431 kept | Established (12/12 is small) |
| 06 Re-solving decomposition | Text-only baselines on WorkspaceBench | 400 items, 8 prompt-only arms | Established |
| 07 Template mining | Native phrasings per feature, frozen before any steering (sha256 0915…923b) | 14 features | Done |
| 08 col pilot + K3 | FollowFeature on col; matched-dose calibration on language and tense | 60 items/feature, pilot split | Pilot |
| 09 OL mechanism | Acting vs reading for OL's continuation channel | pilot split | Pilot |
| 10 Operand gate | Operand digits at "=" (after Kenney): G1–G3, two bases, leak-clean G4, site control | 600 + 117 items; ≈ $7 | Pilot, exploratory |

---

## 4. Results

### 4.1 Part A: what counts as non-workspace content (Q1)

**The standard test is uninformative.** On all eight exp 02 features (col, col_report, bracket depth, predicted POS, tense, language, topic, next token), a retrained probe keeps ≥ 98% of its score after removing the top-384 or top-1,536 J-space, including the in-J references (topic 0.849 → 0.842 / 0.834; language 0.966 → 0.970 / 0.966). Removing ten random subspaces of the same rank gives the same retention within ±0.02. Even at rank 4,000 of 5,120, the excess drop beyond random is at most 8% of the above-chance margin. A strongly represented feature is linearly decodable from almost any subspace of a 5,120-d residual, so the survival test measures probe redundancy, not workspace membership. (First seen in exp 03, where language kept AUROC ≥ 0.998 after J-removal despite being 3–7× J-enriched.)

**The calibrated gate separates them.** In-J references are enriched (topic 5.98 / 2.34, tense 5.60 / 2.31, language 3.32 / 1.73 at r\* / 4r\*) and lens-readable (topic AUROC 0.83, language 0.62, tense 0.52 above band). The count features are J-depleted (col 0.67 / 1.10; col_report 0.59 / 0.86; depth 0.57 / 1.16), and col's value never appears in the lens readout (AUROC 0.500, band [0.499, 0.501]; lens-bag R² −0.03 vs 0.38 from the last four tokens).

**One clean non-J feature: col, characters since the last newline in hard-wrapped prose.** Probe R² 0.655 (control 0.12); non-J under both lenses (camilablank 0.82 / 1.41); causally used: steering the J-removed direction shifts newline log-odds by +0.99 nats at q90 against 0.02 for random directions (49×), no leak at L38; keeps 79% of its full-direction causal effect after J-removal, against 12–23% for every non-count feature (tense 19%, language 5%). `col_report`, the same count under an explicit "how many characters" question, also stays non-J; it was meant to be the in-J count calibration and that calibration failed: Qwen does not pull a count into its J-space on request.

**Everything else fails.** Bracket depth is DIFFUSE-BUT-J-READABLE (the lens shows depth through bracket and quote tokens, R² 0.62). Predicted POS is IN-J through token classes. Tense and language are IN-J. In exp 04, capitalisation (enrichment 2.71, lens AUROC 0.81), in-quote state (2.22) and French gender agreement (fails G1 at 0.60) are IN-J; coding language and words-since-sentence-start are DIFFUSE and leak; indentation level passes G1–G3 (enrichment 1.11 / 1.14, lens unreadable, probe 0.965) but fails G4 (×0.6 random): **decodable-unused**. The list-index count calibration also failed: digits never enter the lens top-10 and the index direction has no causal effect.

**Survey conclusion (~16 gate runs, ~13 features):** on Qwen3.6-27B at L42, the J-depleted, causally used content among natural features is positional bookkeeping, and numbers do not enter the J-space at this site. Stated as a survey, not a theorem.

### 4.2 Part B: does any reader follow col? (Q2, Q3) — pilot

**Calibration passes (K3).** At matched relative norm the NLA tracks language (AUROC 0.87, z +30) and tense (0.92); OL tracks language (0.80, z +20) and, with mined continuation templates, tense (0.99). The J-Lens reads both. So the instrument sees workspace features at the dose used for col.

**No descriptive reading of col.** With fixed descriptive templates, every reader stays inside the random-direction null band on the J-removed 4r\* col direction at q90: NLA AUROC 0.50–0.53, OL 0.49–0.51, every z within ±1.5, for every template family. No free readout mentions line length or position. The same edit moves the model's own behaviour (newline log-odds +0.72, z +55; wrap length −1.55 characters) while the J-Lens does not move (leak-free at 4r\*; the top-384 removal was not enough and leaked EOL tokens, D13).

**Two candidate narrow channels with native phrasings (D24, retracting D21's "neither"):** OL's next-token continuation pair ("- \nthe" vs "- the"), AUROC 0.67–0.72, z ≈ +3.9, requiring the LoRA (collapses adapter-off); and one NLA phrasing ("Mid-sentence continuation located near the end / beginning of a hard-wrapped line"), AUROC 0.63, z +3.5 against a proxy null, best of 11 templates and therefore selection-inflated (pre-registered expectation: shrinks to z ≈ +2.5–2.9 and fails the claim criterion). These are candidates, pre-registered as P2 and P1.

**OL's channel behaves like re-enactment on col (exp 09).** The gain ratio of OL's continuation shift to the model's own behavioural shift at the same edit is 0.99 [0.83, 1.20] for col raw and 1.06 [0.89, 1.26] for col 4r\*, with identical retention under J-removal (0.79 vs 0.75). Random directions with a behavioural effect give 1.65–1.89; tense raw 3.40. But OL also tracks the tense 4r\* direction (z +12 to +16) where the model's next-token tense readout is flat (−0.04 nats), and shows a sub-threshold dose-dependent response to indentation, which the model does not act on. So OL's channel is not a pure re-emission of the model's next-token decision; the binary acting-vs-reading verdict was demoted to three descriptive quantities (rev 4.2) because the test is confounded by context.

**Natural col is text-explicable.** On naturally varying line positions, a text-only Gemini reaches 0.97; the NLA's 0.60 falls to 0.51 after regressing out the text (D23).

### 4.3 Part C: what the readers do read (Q5)

**Tense's tail beyond the top-1,536 J-space.** With fixed templates only the NLA moved (0.60, z +8.7), briefly read as an NLA-over-OL contrast. With mined templates it did not survive: OL 0.66–0.79 (z +12 to +17), NLA 0.59–0.74. Leak-free under label and content sets. Tense is self-reported 87% of the time, so this is the distributed tail of a verbalisable feature, not "what the model can't say". Both cells are main-run primaries (P3, P4).

**Planted intermediates = in-J calibration (exp 05).** Swapping the bridge entity's class-mean projection at L38 ("the country famous for the Eiffel Tower" → Sweden) with the text unchanged: 431/614 candidates pass the keep criteria (answer flips, unrelated prompts unchanged at KL ≤ 0.1, prompt still decodable at L42). All three readers report the planted bridge: OL 1.00, NLA 0.98, J-Lens 1.00, from 1–3% unplanted, z ≥ +84 against random plants. The swap direction is 6.5× J-enriched, so this is calibration, not non-J evidence. Doses were large (median 0.49‖h‖); the main run makes the α ≤ 1.5 subset primary.

**The NLA generates its "prompt" from the latent.** In free explanations of planted activations the NLA names the planted country 12/12 times and rewrites the clue to fit ("the country whose CEO designed adopted Volvo… Sweden"; "maple leaves (Canada)"). Its prompt reconstruction is confabulated to match the latent, which rules out "decode the text, then compute" as its mechanism.

**Operand pilot (exp 10, externally prompted).** On Kenney's 600 "a ± b =" prompts, his probe numbers replicate at L42 (exact operand a 0.825, b 0.932; the NLA reports them 5% / 38%). Operand digits at "=" are J-depleted: class-mean enrichment sits at the shuffled-label reference under both the n1000 and J++ bases, probe-weight and magnitude directions fall below random, and the J-Lens never emits a digit (weak magnitude signal only, OvR 0.52–0.58). The answer is not represented at L42 even with 3-shot prompts where the model is 600/600 correct (probe 0.097). Few-shot context halves operand decodability (a → 0.415). But on 117 one-digit few-shot items, no leak-clean J-removed edit at r11 or r22 changes a single greedy answer (log-odds shifts ≤ 0.1 nats against a 6.65-nat margin); Kenney's class-mean subspace swap is void under the leak rule and flips nothing here; and replacing the entire L42 vector at "=" with a donor's flips 1/117. The "=" position at L42 does not carry the answer-determining operand information, so the zeros are a property of the site, not of the directions. Verdict: **decodable-but-unused at L42**; S10 is not filled; nothing in the main run changed (D33). Convergences with Kenney: readers track represented-but-unused content; his J++ mean-ablation removal AUROC 0.49 matches our survival result; re-imagined prompts are common.

### 4.4 Part D: WorkspaceBench (Q4)

**Replication:** all 12 published numbers (4 regex families × OL / NLA / J-Lens) fall inside our 95% bootstrap CIs. **Protocol asymmetry:** OL's published lead over the NLA comes from scoring the best of 11 layers per item, worth +0.08 to +0.22; at the same layer (42) the NLA ties or beats it.

**Re-solving decomposition (n = 400):** every item either prose reader gets right is also reachable from the prompt text alone (NLA: 0 of 187 hits missed by all text-only arms; OL: 4 of 159, all Chinese-form scoring artefacts). A text-only base Qwen writing in the NLA's register scores 0.46 against the NLA's 0.47 (McNemar p = 0.75). A text-only re-solver given only the NLA's own readout scores more than the NLA (0.57 Qwen / 0.60 Gemini); given the prompt, 0.89; given prompt plus readout, 0.88. Text-verifiable precision of readout claims: NLA 0.62, OL 0.52, text-only 0.92. **Side finding:** the benchmark's frozen prompt-only floors look ≈ 0.14 too low (z +2.2 to +3.3 on 3 of 4 families), cause unknown.

---

## 5. Conclusions drawn

### 5.1 On the title question (pilot-level, pending confirmation)

**Not much, and no.** On Qwen3.6-27B at layer 42, the NLA and Oracle Lens read the same things the J-Lens reads. Both track in-J features at matched dose; both are silent, in their descriptive language, on col, the one certified non-workspace feature the model uses. Their benchmark success is explained by the prompt text. Two exceptions are narrow and channel-specific: OL responds to col only through its next-token continuation, moving one-for-one with the model's own newline decision; the NLA responds in one phrasing that is expected to shrink on fresh items. Where the NLA does report content the lenses miss, it is content the model represents but does not use at that site (operands at "=", indentation), and the NLA's prompt reconstruction is generated from the latent rather than read from the text.

Pre-registered headline, likeliest branch **1b: "Not the NLA, and OL only through its next-token continuation channel."** Alternatives: 1a (both readers weakly sensitive in narrow channels, "channel-specific sensitivity", never "reads"); 1c (only the NLA phrasing replicates); 2 (neither reads the non-J feature, and the null is interpretable because both read workspace features at the same dose).

### 5.2 Conclusions that stand on every branch

1. **The standard "probe survives J-removal" test cannot identify non-workspace content.** It passes everything, in-J references included, by the same margin as random removal. Any paper using it as evidence of "not in the workspace" should be read with that in mind.
2. **A calibrated, nulled gate can certify non-J content, and on this model it finds positional bookkeeping.** col is non-J under two lenses, causally used, and keeps 79% of its effect after J-removal. Numbers never enter Qwen's J-space at L42, so there is no in-J count control; this is stated as a scope limit.
3. **WorkspaceBench, as scored, cannot separate reading from re-solving.** The prompt alone outscores the readouts; adding the readouts to the prompt does not help. OL's published lead is a best-of-11 effect.
4. **The NLA generates rather than re-solves.** Under a planted bridge it rewrites the clue 12/12 times. This is robust to which branch the main run lands on.
5. **Method lessons that transfer:** additive steering produces dilution artefacts when h·v ≠ 0 (use centred steering); a two-sided leak detector is needed to certify any "non-J" dose; the null band, not n, limits power; a matched-dose positive control is required before any null is interpretable; templates must be mined from the reader's own output distribution and frozen before steering.

### 5.3 What is still open

- Whether P1–P4 survive Holm at z ≥ 3 on fresh items (19–21 October).
- Lens-dependence of "non-J": two lenses agree on col, but a third (J++) was only used for operands. The J-space has no spectral elbow.
- Where Kenney's 28% answer flips come from (site or layer set): a causal test of operand use needs edits at the operand positions or multi-layer patches.
- Why few-shot context halves operand decodability.
- OL on tense: it tracks a direction the model's next-token readout ignores. Whether a generation-level tense measure closes that gap is S6/S7.
- A second non-J feature that is not a count, on another model or layer, would strengthen the survey conclusion considerably.
- Power: nulls at n = 60 with K = 20 bands are suggestive only; the main run uses K = 100.

### 5.4 Retractions carried into the record

"Neither reader reads col" (D21, overstated: two narrow channels exist); "NLA > OL on tense 4r\*" (template artefact); "digits are almost certainly in J-space" (wrong: operands at "=" are J-depleted under two bases); "the NLA drops in-workspace specifics" (overstated: the probe's lead rides partly on decodable-unused directions); and a 2× underestimate of exp 09's cost. All logged in `notes/decisions.md`.
