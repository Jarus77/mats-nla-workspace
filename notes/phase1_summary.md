# Phase 1 mini-project summary (source: `mini-project.pdf`, 19 pages, read in full)

## 0. What the document actually is

- **Title:** "In-context introspective detection of J-space contents in Qwen3-8B".
- **What kind of work:** a concept-injection introspection study, in the style of Anthropic's "injected thoughts" experiments, done on an open model. It uses the **Jacobian lens (J-lens, Gurnee et al., served by Neuronpedia)** as the independent ground-truth instrument for "what is in the residual stream / workspace".
- **What it does not do:** it does not compare NLAs or Oracle Lens. It uses no linear probes trained on residuals. "Global workspace" appears only as the **"workspace band"** (the layers where the J-lens becomes readable).
- **Compute:** one 24 GB NVIDIA L4, about 2 h for everything. 19,700 trial rows in total.
- **Code was published (possibly recoverable):** https://anonymous.4open.science/r/jspace-selfreport-0586/
  - Contents: prompts, battery, passages, injection + readout code, every trial as JSONL (19,700 rows), and figure scripts. Each figure script ends with an independent recomputation of every plotted number.
  - Check this link first before re-implementing. Anonymous-4open repos often expire, but may still be live.
- **Paths referenced:** `paper_figs/fig1.png` … `fig9.png`.

## 1. Project, research question, main claims

**Model and lens**
- Qwen3-8B (36 layers, d = 4096), bf16.
- Chat template, thinking disabled, so the answer begins at a fixed offset.
- Lens: Neuronpedia's fitted Jacobian lens, 35 matrices J_l (used unchanged). Ranks are 0-based.

**Question.** Can an open 8B model tell you what is in its own residual stream when asked in plain language? If self-report is trustworthy, it is the cheapest auditing readout available. If not, what does it measure instead?

**Pre-registered hypotheses**
- **H1:** the model reports an injected concept above a matched clean baseline.
- **H2:** the report is about the *concept*, not about *having been perturbed*.
- **H3:** the report tracks what the lens says is present.

**Main claims, with numbers**

1. **H1 holds.**
   - Mean P(yes) rises from 0.012 (clean) to 0.833 at gamma = 0.4.
   - 99.3% of concept trials are above 0.5 at the top dose.
   - The paired difference from clean excludes zero at every dose (widest lower bound 0.017 at gamma 0.05, 0.79 at gamma 0.4).
2. **H2 fails for the yes/no channel. It is a disturbance detector.**
   - Mean P(yes) at gamma = 0.4: concept 0.830 vs **J-space random direction 0.792** (gap 0.039, CI 0.02-0.06 passage-resampled, 0.01-0.07 concept-resampled) vs isotropic random 0.381 vs clean 0.012.
   - At gamma 0.2: concept 0.764, J-random 0.663, isotropic 0.112. "The disturbance detector is a curve, not a point."
   - Per-concept P(yes) ranges from 0.61 to 0.98 (ICC 0.71). This is direction-dependent but not concept-selective, consistent with Macar et al. and Hahami et al.
3. **H2 holds for the naming channel.**
   - 216/1,000 concept trials at gamma 0.4 name the concept in the top 5. CI 0.14-0.31 concept-resampled.
   - 0 named across 6,000 isotropic-random trials and 0 clean.
   - Cross-talk: a non-injected battery concept reaches the top 5 on 8 of 24,500 concept×slot pairs (0.03%), all same-category neighbours: `bread` under `cheese` ×2 and under `pasta` ×4, `grief` under `sadness` ×2.
   - Caveat: in the J-random arm the random token itself reaches the top 5 on about 2% (10/500) of trials, so naming "reads whatever direction was written".
4. **H3 fails as pre-registered.**
   - Ground truth was read at the last user-turn token ("summary token"), layer 22. There the lens predicts naming at **AUC 0.556**.
   - At the **answer slot** in the same forward pass, AUC is **0.761** (under a neutral question).
   - Yes/no does not track the lens at all: P(yes) is 0.826 on landed trials vs 0.836 on not-landed trials.
5. **The hand-off happens at layer 23.**
   - The concept is erased at the summary token and pulled into the answer slot from layer 23.
   - Pooled answer-slot median rank: 670 (L22), 14 (L23), 5 (L24).
   - User-token median goes from 57 to 833 between L22 and L28.
   - Answer-slot AUC: 0.64 (L22), 0.73 (L23), 0.81 (L25), against 0.60-0.65 at the user token.
6. **Patching: the answer-slot residual is necessary from L23 but not sufficient until very late.**
   - Necessity kill rate (clean → injected run, answer slot): 12% at L16, 49% at L22, 94% at L23, 100% from L24.
   - Sufficiency (named source → clean run, answer slot): 0 of 111 named sources transfer at L24, even though rank moves from 7,754 to 98. Transfer is 46% at L30 and 85% at L34, which is inside the circular band.
   - Overwriting the user-turn positions instead still kills naming late: 95% at L24, 62% at L28, 30% at L32. Between L23 and L30 the name is assembled from the answer-slot residual plus continued attention back to the injection.
   - Ablating only v_c (rank-1 projection) at L24 kills naming on 90% of named sources (95% if the projection is set to the clean-run value). A random J-space direction kills 0-3%. Clean runs sit at about -5 along v_c, so zero-projection under-ablates.
   - Emotions commit earliest (31% killed at L28), countries latest (84%).
7. **Colours are never named.** No injected colour vector produces a name at any dose, and no patch at any layer makes the model name a colour. Separately, on a natural prompt the model names `yellow` (its top token, rank 0) while the lens has yellow at rank 19,812.
8. **Natural-prompt stratum** (for any follow-up training):
   - Requiring lens agreement at L25 **and** L26 leaves 3 of 109 prompts mislabelled absent and discards 113 of 300.
   - A rewrite of the weakest 38 concepts (228 new descriptions) made it worse: only 2 of those concepts ended up present.

**Author's stated next run:** the main matrix again with ground truth at the answer slot, stratified by category, with the J-space random arm at every dose. "Pre-registering the summary token and the isotropic control were the two decisions that cost most."

## 2. Concept sets, and where naming failed or succeeded

### Battery
- 50 leading-space single-token concepts, 10 each in **countries, animals, foods, colours, emotions**.
- Drafted from about 120 Claude-generated candidates, then checked against the Qwen3 tokenizer for single-token status with a leading space.
- Hand-filtered for polysemy and frequency.
- **The PDF does NOT list all 50.** Only these appear by name:
  - Countries: `France`, `Canada`, `Brazil`, `Japan`, `Egypt`
  - Animals: `whale`, `spider` (`whales` appears in the scorer-bug note)
  - Foods: `chocolate`, `bread`, `cheese`, `pasta`, `honey`
  - Colours: `purple`, `navy`, `brown`, `yellow`
  - Emotions: `boredom`, `guilt`, `grief`, `sadness`
- **Excluded candidates:**
  - Not single-token: `maroon`, `scarlet`
  - Polysemous across categories: `orange`, `turkey`, `china`, `pride`, `mouse`
  - Too frequent (clean rank near threshold): `love`, `red`, `white`, `gold`
- **Flagged leak:** `navy` draws ` ship` and ` military` into the answer slot. It was flagged, not swapped.
- **Held-out split** (for Phase 2): 2 concepts per category, seeded sampler. Four of them have no natural-present example: `Canada`, `honey`, `brown`, `yellow`.

### Carrier passages
- 20 administrative passages of 13-18 tokens.
- Rules: administrative register, no place names, nothing living or edible, no colour words, no weather or seasons, no verbs of judgement.
- Clean filter: no concept in the 50×50 clean matrix scored below rank 100. The best concept median is 20,537 and the lowest cell is 79,950, so nothing was dropped.
- This is the **input-inferability control**: the concept cannot be inferred from the text.

### How success was measured
- **"Named":** after prefilling yes, rank of c < 5 (top 5) at the CONCEPT slot, with the answer-token-mass < 0.5 exclusion (incoherent slots dropped).
  - Rate depends on cutoff at gamma 0.4: 5.6% top-1, 21.6% top-5, 38.7% top-10.
- **"Landed":** J-lens rank ≤ 25 at layer 22 at the last user-turn token (pre-registered).
- **"Present at answer slot":** lens rank ≤ 25 at the answer slot at some layer. For natural prompts, "present" is rank ≤ 100 and "absent" is rank ≥ 1000.

### Per-category results at gamma = 0.4 (200 trials per category)

| category | landed | named | named if landed | named if not | AUC |
|---|---|---|---|---|---|
| countries | 0.84 (168) | 0.15 (30) | 0.18 (30/168) | 0.00 (0/32) | 0.73 |
| animals | 0.23 (46) | 0.26 (51) | 0.76 (35/46) | 0.10 (16/154) | 0.89 |
| foods | 0.20 (40) | 0.43 (85) | 0.50 (20/40) | 0.41 (65/160) | 0.43 |
| colours | 0.27 (53) | **0.00 (0)** | 0.00 (0/53) | 0.00 (0/147) | undef |
| emotions | **0.00 (0)** | 0.25 (50) | undef | 0.25 (50/200) | undef |

### Named concepts: verbalization FAILED vs SUCCEEDED (verbatim from the PDF)

**Failed (lens-present, not named):**
- **`purple`**: lands on 20/20 passages, named on 0.
- **`navy`**: lands on 17/20, named on 0.
- **All colours**: 0/200 named. Colours supply 53 of the 307 landed trials, all at 0% naming. They never fall below median rank 22.5 at the answer slot, and only 31/50 colour trials reach rank 25 at the answer slot at any layer. The colour patch leaves median rank 104 at L34.
- **`Brazil`, `Canada`**: land on about 100% of passages but are named only about 5% of the time (Fig 3/5).
- Countries overall: 84% landed vs 15% named. Partly this is capitalisation: removing the "single lowercase noun" constraint raises naming from 0.150 to 0.225 (+0.075, CI 0.03-0.13), which recovers about a third of the stall.
- **16 concepts neither land nor are named.** These are not listed by name.

**Succeeded (named although never landed at the user token):**
- **`boredom`**: lands on 0, named on 20/20.
- **`bread`**: named about 0.75, landed about 0.
- **`guilt`**: named about 0.5, landed about 0.
- More generally, emotions never enter the top 25 at the user token at any layer (best rank 30), yet they are named on 25% of trials. At the answer slot they are present on 42/50 (84%). This is a position artifact, not a non-verbal representation.

**Both landed and named:**
- **`whale`**: lands 20/20, named 17/20.
- **`chocolate`**: lands 20/20, named 20/20.
- **`spider`**: about 0.55 landed, 0.84 named.
- `France`: landed about 1.0, named about 0.3.

### Natural (unaided) naming
- 50 two-hop prompts ("the eight-legged creature that spins silk webs") were named unaided on 35/50 (neutral wording) and 41/50 (injection wording).
- Round 2 rewrites named 24% vs 37% in round 1, with **countries at 2%**. Countries are named unaided on about 10%. "I have no description style that works for countries yet."
- On round-2 failures, the answer-slot top 5 is `triumph`, `symbolism`, `legacy`, `heritage`: the model reads a list of proper nouns as a request for a theme.

## 3. Methods (enough to re-implement)

**Lens and injection**
- Lens readout: softmax(W_U · norm(J_l h)). J_l reads the residual leaving block l.
- Injection vector: **v_c = J_l^T (g ⊙ W_U[c])**, where g is the final RMSNorm gain. It is the adjoint of the lens map applied to c's unembedding row.
- Sanity identity: ⟨v_c, h⟩ equals the lens logit for c times RMS(h), to 5 digits. Using J instead of J^T breaks this by a factor of about 270.
- Normalise to unit length and scale to **alpha_l = gamma · median‖h_l‖** over user-turn positions (61.6, 68.3, 70.9 at L12, L13, L14).
- Add to the residual entering blocks **12-14**, at user-turn positions.
- Doses: gamma ∈ {0.05, 0.1, 0.2, 0.4} were pre-registered; 0.6 and 0.8 were added post hoc. **Only gamma = 0.4 is a coherent, useful dose.**
  - Above 0.4 the answer slot breaks: mass 0.350 at 0.6 and 0.156 at 0.8, and the exclusion removes 875/1000 trials at 0.6.
  - Even at 0.4: minimum mass 0.594, mean 0.9345, 349/1000 trials below 0.935. Countries fail first (mean 0.853).

**Read layer and two-hop check**
- Read layer 22 was a "judgement call" taken at the leading edge of the top-1 accuracy rise.
- J-lens top-1 next-token accuracy stays below 0.01 through L21 and reaches 0.27 at L34. The logit lens is at 0.02-0.03 from L18 and reaches 0.30; the two are within 0.03 everywhere.
- The pre-registered excess-kurtosis step criterion failed (0.84-1.64 over L4-27).
- Two-hop justification: on "the animal that spins webs", the J-lens ranks `spider` 1st at L11 while the logit lens ranks it 79th.

**Elicitation**
- Version 3 was the only one that validated. The exact prompt text is **not quoted in the PDF**; it is in the repo.
- Rules learned:
  - Remove every line the model could echo. v1 said yes on 20/20 clean and copied the passage into the CONCEPT field. v2 copied the instruction line (clean P(yes) 0.62). v3 gave clean P(yes) 0.012 over 20 passages (0.029 on the pilot passage).
  - Include a symmetric prior and a prefill.
  - The prompt must contain no text that can stand in for an answer.
- Readout is from logits, because sampling failed on format:
  - Prefill `INJECTED:`, then P(yes) = mass(yes variants) / (mass(yes) + mass(no)) over cased variants (forced choice).
  - Then prefill the yes and take the rank of c at the CONCEPT slot.
- Mass exclusion: answer-token mass < 0.5 means the trial is dropped. The 0.5 threshold was fixed in advance.
- Free generation is much worse than teacher forcing: 30/500 (6.0%) vs 36/100 (36%) on passages 0-1. It never "blurts" (0/500 mention the concept before the CONCEPT field), and says yes on 87.4% of generations.

**Controls**
- **Clean:** mean P(yes) 0.012, 0/1000 above 0.5, 0 named.
- **Isotropic random:** a seeded random unit vector per layer at the same alpha_l. It has a cosine of about 1/64 with any fixed direction, so it controls only for norm. Its P(yes) is bimodal (24.1% of directions > 0.9, 48.8% < 0.1). Best target rank across 4,000 trials is 45.
- **J-space random (the right control):** the lens direction v_tok of a random non-battery vocabulary token ("junk by design": ` Suff`, ` onDelete`, ` uintptr`), built exactly as for a concept. 500 trials at gamma 0.4 on passages 0-9; 496 distinct strings; not bimodal (0.2% below 0.1); mean mass 0.943.
- **Input-inferability:** the passage rules above, plus the clean-rank filter.
- **Cross-concept specificity:** every battery concept is scored in every injection's slot.
- **Patching controls:** a clean-to-clean patch leaves rank at 7,754 with 0 named. Non-naming sources transfer at 0% through L30.
- **Second model:** Qwen3-4B has a clean floor of P(yes) 0.247 (25% of clean trials > 0.5), so it is unusable as an injection subject.

**Statistics**
- 95% bootstrap intervals, resampling over the 50 concepts by default, because naming clusters by concept (ICC 0.51).
- Passage-clustered intervals are used for the H1 lower bounds.
- Design: 9,000 pre-registered trials (4 doses × 50 concepts × 20 passages × 2 arms + clean). 4,000 post-hoc trials at gamma 0.6/0.8 and 5,700 control-arm trials in Sections 3-6 bring the total to 19,700.

**Natural prompts**
- 300 descriptions. A sweep of L20-32 at the answer slot found no layer that is both clean and non-circular:
  - L24: 58 present (rank < 100), 44 named-but-lens-absent, correlation with own answer 0.65.
  - L26: 122 present, 3 named-but-absent, correlation 0.80.
- Two-layer rule (L25 and L26 agree): 3/109 mislabelled absent, 113/300 discarded, AUC 0.76 either way. 691 injected trials are present under it vs 721 at L24 alone.
- Wording moves the lens: concept at rank ≤ 100 at L24 on 35/50 under "what concept was injected" vs 22/50 under "what is this passage about". `yellow` sits at rank 4,527 vs 19,812 under the two wordings.

## 4. Weaknesses

### Stated by the author
- One injection subject (Qwen3-8B), one coherent dose (gamma 0.4), and one read position in the main run.
- The J-random control is one dose on 10 passages.
- The answer-slot result is 250 trials and is **partly circular from L23**. The answer-slot lens vs output log-rank correlation is 0.43 (L22), 0.59 (L23), 0.69 (L24), 0.77-0.94 (L29-34). Nothing from L29 up is used.
- Three contrasts fail to exclude zero under concept resampling: the pooled landed-vs-not naming difference (CI -0.08 to 0.28), pairwise category contrasts (except vs colours), and the foods slope.
- `navy` is polysemous. The country effect is confounded with capitalisation, unembedding norm and lens cosine.
- Rows don't reproduce across bf16 reruns. Aggregates do (0.8302 vs 0.8306), but answer rank matches on only 229/500 and naming flips on 7/500.
- Nothing shows the yes/no answer is more than a readout of how far the answer-slot distribution was moved.
- The concept-vs-noise three-way prompts never validated: zero mass on `noise` when clean, and mass on the three options fell to 0.095 under concept injection vs 1.000 under isotropic. That cost 3,500 trials and taught one fact: a concept direction breaks a multi-word slot where an isotropic direction of the same norm does not.
- The lens was fit on wikitext, so it may be unable to read `<|im_end|>` or chat tokens at all. The concept fades at the summary token (median rank 21 at L22 → 682 at L34) but `<|im_end|>` does not replace it (73,100 → 3,470).
- Two errors were caught in review:
  - The free-gen scorer was case-sensitive (3.8% → 6.0%).
  - The "minimum mass 0.935" claim was wrong: 0.9345 is the mean, the minimum is 0.594.

### Additional critique (reviewer-style)
- **The "ground truth" is itself a lens, not a causal or probe measure.** "Landed" means J-lens rank ≤ 25. The colour result (lens-present, never named) and the `yellow` result (named, lens rank 19,812) show the J-lens and verbalization disagree in **both** directions. The J-lens is not a neutral arbiter of "in workspace".
- **Injection ≠ natural representation.** Injected presence is about 10× stronger in the lens than natural presence, and the steering direction is built from the lens itself (v_c = J^T g W_U[c]). So "lens sees it" is partly by construction: it is a self-consistency check, not independent validation.
- **Naming is a forced, prefilled top-5 logit rank, not free verbalization.** With a 21.6% top-5 rate and 5.6% top-1 rate, "succeeded" is a weak bar. Free generation gives only 6%.
- **Coherence and dose:** the useful window is one dose wide, and at that dose 35% of trials have degraded mass. Mass and P(yes) correlate at 0.32, which was not removed.
- **Category confounds dominate:**
  - Category dummies explain 41.1% of landing variance; adding the cosine of v_c at L13 with the L22 lens direction adds only 0.3%.
  - The cosine–landing Spearman is 0.45 raw but -0.10 after removing category means.
  - The only per-concept regularity is the pre-normalisation ‖v_c‖ predicting median lens rank (Spearman -0.34, p = 0.015).
  - With 10 concepts per category, "colours are never named" is n = 10 concepts. It could be a tokenizer or format effect (adjective vs "noun" request), not non-verbalizability.
- **No linear-probe baseline.** The project never asks whether information is *linearly decodable* yet *not reportable*. That is the property the new out-of-workspace set needs.
- Many post-hoc arms (0.6/0.8 doses, J-random, patching, natural prompts) were added after seeing results. They are honestly labelled, but the strongest conclusions (answer-slot AUC 0.761, layer-23 step) are post hoc.

## 5. Reuse for an out-of-workspace set on Qwen3.6-27B, and design lessons

### Directly reusable
1. **The repo.** Try https://anonymous.4open.science/r/jspace-selfreport-0586/ first. It has the prompts, the battery, the 20 passages, injection and readout code, and 19,700 JSONL rows.
2. **The injection recipe.** v_c = J_l^T (g ⊙ W_U[c]), unit-normalised, scaled to gamma · median‖h_l‖, plus the ⟨v_c,h⟩ = logit·RMS identity check. This ports to any model with a fitted J-lens. Check Neuronpedia coverage for Qwen3.6-27B; otherwise the lens must be fit.
3. **The control structure:**
   - clean
   - isotropic random (norm-only)
   - **J-space random direction** (the right "disturbance without content" control)
   - cross-concept slot scoring
   - an input-inferability filter: passages stripped of category cues, with every concept's clean rank required to be ≥ 100 (observed ≥ 20,537)
4. **The elicitation rules:** nothing echoable, a prefill, logit-based forced choice, and an answer-mass exclusion (< 0.5) fixed in advance.
5. **The ready-made dissociation candidates** (the seed of an out-of-workspace set):
   - **Colours** (`purple` 20/20 landed / 0 named, `navy` 17/20 / 0, all colours 0/200 named and never named by any patch). These are the best candidate "present-but-not-verbalizable" concepts.
   - **Countries:** `Brazil`, `Canada` (about 100% landed, about 5% named).
   - Conversely, `boredom`/`bread`/`guilt` are "named but not lens-present at the user token". This is a position artifact (emotions reach 84% at the answer slot), **not** an out-of-workspace case. Don't use them as such.
6. **The patching protocol.** Necessity patches (answer slot vs user span, and rank-1 v_c ablation with a random J-direction control) give a causal label, not just a lens label.

### Lessons that should change the new design
1. **Define "out-of-workspace" with a linear probe trained on natural, non-injected residuals, not with a lens.** The J-lens disagrees with verbalization in both directions. If the hypothesis is "J-Lens/Oracle Lens read the workspace, NLAs read more", the ground truth for "decodable" must be independent of all three methods: probe accuracy on held-out prompts with a shuffled-label baseline. "Not verbalizable" must be measured by behaviour: free generation plus forced choice, across several wordings.
2. **Pre-register the read position and avoid the summary token.** The concept moves from the last user token to the answer slot at L23 of 36 (about 64% depth). On 27B, find the analogous hand-off layer and position first (a pilot sweep), then freeze it. Report both positions.
3. **Avoid circular layers.** Answer-slot lens readings from about 80% depth onward (L29+/36) mostly re-describe the output. Compute the lens-vs-own-answer correlation per layer and exclude layers above about 0.78.
4. **Use the J-space-random control, not isotropic.** Isotropic noise overstated concept selectivity by about 10× (0.45 vs 0.039 gap).
5. **Do not rely on yes/no detection.** It reports disturbance, not content. Use naming (forced choice + free generation) as the verbalization measure, and report multiple cutoffs (top-1/5/10).
6. **Expect a narrow dose window.** Sweep doses on 27B with the J-random arm at every dose, and track answer-slot coherence (mass) per category.
7. **Control category confounds:**
   - Use ≥ 20 concepts per category.
   - Match the tokenizer case, unembedding norm and part of speech. Colours are adjectives and countries are capitalised, yet the prompt asked for "a single lowercase noun". Test whether colour non-naming survives a prompt asking for "a colour/adjective".
   - Bootstrap over concepts (ICC about 0.5).
8. **Natural presence is about 10× weaker than injected presence.** For an out-of-workspace set, prefer **natural non-verbalizable information** over injected concepts. Candidates:
   - properties linearly decodable from residuals that models are known to fail to self-report: token position/count, tokenization facts, language-ID of a single token, numeric magnitudes, syntax depth, factual-recall confidence
   - features the model represents but answers wrongly about when asked
   This avoids the "lens sees it by construction" circularity of J^T-built injections.
9. **Prompt wording moves the lens and naming** (35/50 vs 22/50 present; `yellow` rank 4,527 vs 19,812). Fix the question wording in advance and use several paraphrases.
10. **Countries need a different description style** (2-10% unaided naming). Proper-noun clue lists are read as "theme" requests.
11. **Hygiene:**
    - Expect row-level non-reproducibility in bf16 (7/500 flips); report aggregates with CIs.
    - Hand-check scorers (case sensitivity).
    - Save every trial as JSONL along with recomputation scripts, as the Phase 1 repo did. Also keep the code somewhere durable this time.
