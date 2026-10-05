# Literature review and novelty check: do NLAs read non-workspace content that J-lens / Oracle Lens miss?

Compiled 2026-10-05 for a MATS 12.0 (Neel Nanda stream) sprint. Target setup: Qwen3.6-27B, layer 42.

**What "read" and "inferred" mean here.**
- **[READ]** I read the full text. LessWrong posts came through the GraphQL API as full HTML. The two transformer-circuits papers were downloaded as full HTML and grepped. For the WorkspaceBench figures I downloaded the images and looked at them.
- **[SUMMARY]** I only saw a WebFetch or WebSearch summary (abstract level). Check these before citing.
- **[INFERRED]** My own reasoning, not a claim made by the source.
- **WorkspaceBench numbers.** The post reports per-method results **only as a bar chart; there is no table**. Every WorkspaceBench number below is **read off the chart by eye (about ±0.03)**. Get exact values from the repo (https://github.com/camilablank/workspace-bench) before quoting any of them.

---

## 0. TL;DR novelty verdict

**Partially done. The headline question has been attacked from three directions, but nobody has run the clean test on natural prompts.**

**Done, in three ways:**

1. **Synthetic injection, different model, no Oracle Lens.**
   - Source: Viswanath, "Models are blind outside the J-space. NLAs aren't." (LW, 2026-07-08). Llama-3.3-70B, L53.
   - Concept vectors are split into a J part and a non-J part (rank r=2000) and injected one at a time.
   - The model names the J part 80% of the time and the non-J part 0%. The NLA names both 100%.
   - In "Franken-vectors" (A's non-J part plus B's J part), the model names only B, in 3/12 trials, and names A in 0/12. The NLA names both in 12/12.
   - In the line-counting task (natural prompts), **the NLA does NOT read the exact character count that the J-lens shows.**
2. **Reconstruction geometry, on your exact model.**
   - Source: Hong, "Fixing rewards for NLA to reduce confabulation" (LW, 2026-07-24). Qwen3.6-27B.
   - "About 95% of training gains live outside the J-space." The inside-J part of reconstruction is "nearly saturated from the start," and the outside part "almost tripled."
   - This shows the NLA's *extra FVE* is non-J. It does **not** show that the NLA's *text* conveys interpretable non-J content. Hong also finds about 90% of specific claims unsupported.
3. **A concurrent pilot on your exact question, model and benchmark.**
   - Source: `LittleMeHere/nla-vs-lenses` (GitHub, created 2026-10-04, labelled "MATS 12 mini project"). Its first line quotes Neel's doc: "What can NLAs capture that Oracle Lens / J Lens don't? Is it really the stuff not in the workspace?"
   - Finding: on multihop, the prose readers mostly **re-solve** the question from the prompt they decode. On chain arithmetic at L56 the Oracle Lens genuinely reads the number and the NLA is near chance. "On no task does the NLA find a hidden step the oracle lens misses."
   - They write explicitly: **"Not tested: content known to be outside J-space. So 'is it the stuff not in the workspace?' is not answered directly."** Their item #2 under Next is to inject a split J / non-J concept into all three readers.
   - **This is a direct overlap risk.** It is not the user's own repo: the local `notes/phase1_summary.md` describes a different, Qwen3-8B introspection project. Treat it as a concurrent fellow whose next step is the obvious one.

**Still open:**
- (a) Natural (non-injected) activations on Qwen3.6-27B, where the ground truth is a J-orthogonalized probe, in categories the J-space paper shows are used but not verbalized: part-of-speech, tense, linewrap character counts. All three readers at L42, plus a text-only baseline. **Nobody has done this.**
- (b) The injection split run with **Oracle Lens** as a third reader. Nobody has done this. Viswanath ran NLA versus model self-report only, on Llama.
- (c) Whether NLA claims that are absent from J-lens and Oracle Lens are (i) prompt restatement, (ii) confabulation, or (iii) verifiable non-workspace content. nla-vs-lenses has a first-pass breakdown for multihop only, and none of its claims are checked against a probe.

---

## 1. Sources

### 1.1 WorkspaceBench (Blank et al., LW, 2026-09-23) [READ, figures viewed]
https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the · repo https://github.com/camilablank/workspace-bench

**Scope**
- 3,356 questions in 27 eval families, built for **Qwen-3.6-27B** (64 layers). Done with MATS support. Thanks to Jack Lindsey and Dan Mossing.

**How "workspace" is operationalized.** A WorkspaceBench intermediate must satisfy:
- The model gets the answer right >80% of the time over n=10 rollouts at temperature 0.8, with no chain of thought.
- The intermediate never appears in the prompt as a numeral or a word.
- The model never states it in its continuation.
- There is a single solution path, checked by sampling.
- Quote: "we follow the general intuition that sufficiently complex computation could surface important variables to be read... an intermediate... can be expected in the workspace if the model computes it in a single forward pass and uses it."
- The "ideal workspace reader" is defined as "a strict improvement over J-lens... a 'multi-token J-lens'", plus other useful intermediates, plus few hallucinations.

**Readout protocol.** This matters for fairness.
- J-lens, R-lens and logit lens: top-10 tokens at **L20–60, every 4 layers** (11 layers).
- Template lens: same layers.
- Oracle Lens: 1 sample of 4 bullets (about 90 tokens) at each of the same 11 layers.
- **NLA: 1 sample (about 250 tokens) at L42 only.**
- A pass at **any** (layer, position) cell counts as a pass. **[INFERRED]** This favours methods read at 11 layers over the NLA's single layer.
- For many evals, single-token readouts are first turned into concepts by a prompt-blind summarizer LLM.
- Judges: Gemini-3.8-Flash for most evals, Sonnet-5 for safety evals. The "baseline evals" use regex.

**Baselines**
- *Lucky guess*: answering multiple choice without the question.
- *Prompt-only*: an LLM reads the prefix up to the read position and answers "what would a model be thinking of at this point?". The authors call this "an upper floor for what can be inferred from the input."
- The authors admit that "Many of the tasks... were selected precisely because we logically inferred from the prompt text alone what the workspace may contain."
- On text inversion: "a tool that simply inverts activations back into the input text... lets an agent infer an intermediate from the prompt rather than from the activations. Our benchmark doesn't fully rule out shortcuts like this."

**Hallucination eval**
- Qwen answers assistant prompts.
- Headline: "fraction of readouts with at least one wrong claim (lower is better)", judged against Qwen's own response.
- There is also concept precision and recall against the J-lens top-10.

**Per-method pass rates (read off the chart; order OL-RL / OL-SFT / J / R / Logit / Tuned / NLA-RL / NLA-SFT / SAE / prompt-only):**

| eval | OL-RL | OL-SFT | J | R | Logit | Tuned | NLA-RL | NLA-SFT | SAE | prompt-only |
|---|---|---|---|---|---|---|---|---|---|---|
| basic readout | .99 | .95 | .90 | .89 | .84 | .92 | .70 | .48 | .20 | .72 |
| multihop | .99 | .98 | .82 | .58 | .45 | .62 | .75 | .47 | .32 | .90 |
| multilingual | 1.0 | .99 | .96 | .93 | .93 | .95 | .81 | .59 | .27 | .76 |
| typo | 1.0 | 1.0 | .61 | .83 | .49 | .85 | .98 | .94 | .96 | 1.0 |
| typo mt | .83 | .84 | .17 | .25 | .20 | .20 | .75 | .42 | .16 | .46 |
| multilingual typo | .78 | .63 | .05 | ~0 | .02 | .03 | .60 | .04 | .06 | .48 |
| poetry | .55 | .33 | .07 | .02 | .06 | .06 | .21 | .05 | .01 | .71 |
| directed modulation | .28 | .28 | .09 | .07 | .06 | .03 | **.05** | ~0 | .02 | .21 |
| multi-concept dir. mod. | .32 | .32 | .12 | — | — | .04 | .12 | .04 | — | (lucky .36) |
| arithmetic intermediates | .34 | .31 | .01 | .01 | .01 | .01 | .24 | .22 | — | .15 |
| chain intermediates | .25 | .25 | .22 | .19 | .17 | .21 | **.03** | .02 | — | .03 |
| brew intermediates | .70 | .73 | .86 | .76 | .43 | .31 | .66 | .50 | ~0 | .96 |
| buggy code | .29 | .27 | .22 | .22 | .22 | .22 | .25 | .23 | .22 | .04 |
| association | .87 | .76 | .39 | .40 | .17 | .21 | .65 | .30 | .36 | .29 |
| conjunctive association | .14 | .07 | .01 | .02 | ~0 | .02 | .04 | ~0 | — | .07 (lucky .11) |
| user modeling | .94 | .97 | .76 | .58 | .57 | .77 | .72 | .43 | .39 | .07 |
| role-bound association | .63 | .63 | .13 | .05 | .03 | .06 | .40 | .04 | .01 | 1.0 |
| relational multihop | .50 | .39 | .01 | .04 | ~0 | .01 | .37 | .02 | — | .82 |
| moral rationale | .76 | .62 | .27 | .26 | .19 | .28 | .45 | .07 | .20 | .49 |
| agentic misalignment | .58 | .63 | .54 | .51 | .27 | .35 | .53 | .02 | .02 | — |
| jailbreak recognition | .93 | .95 | .87 | .81 | .77 | .80 | .88 | .63 | .62 | — |
| **hallucination (lower better)** | **.34** | **.28** | **.31** | **.21** | **.20** | **.27** | **.96** | **.97** | **.36** | — |

Other hallucination values: Template lens ≈.40, Patchscopes ≈.06. "mt" means the multi-token variant.

**J-lens precision and recall chart (metamodels only, read off the chart)**
- NLA-RL L42: recall@10 ≈ 0.50, precision ≈ 0.24.
- NLA-SFT L42: ≈ 0.47 / 0.21.
- OL-RL L44: ≈ 0.49 / 0.29. OL-SFT L44: ≈ 0.53 / 0.33. OL-SFT L40 is best at ≈ 0.57 / 0.36.
- The authors' caveat: "lower precision with respect to J-lens doesn't necessarily imply hallucination. It can also mean multi-token content the J-lens can't express."

**Stated conclusions**
- "single token lenses are a bag-of-words... no way of encoding order" (examples: multi-digit numbers, "thief chases policeman").
- "NLAs... happen to surface a lot of workspace content! ... average readout being roughly twice the length of the average Oracle-lens readout... nearly every NLA readout contains some type of claim that is directly contradicted by the surrounding context."
- "if the verifiable claims... are largely false, we put low confidence on the trustworthiness of the unverifiable claims."

**Implications for the sprint [INFERRED]**
- Where NLA-RL beats J-lens by the most, prompt-only is also strong: multilingual typo, typo mt, relational multihop, role-bound. That fits NLA (and OL) recovering prompt content and re-solving the question.
- On the items least inferable from surface text, the NLA is weak:
  - Directed modulation: NLA .05 versus OL .28. The judge excludes readouts that only describe the instruction.
  - Chain intermediates: NLA .03 versus J .22.
- **WorkspaceBench contains no eval targeting non-workspace content by design.** It cannot answer Neel's question as built.

**Appendix details**
- Oracle Lens training (as stated in the post):
  - An AR is trained on on-policy spans, using MSE in whitened space on 50M tokens.
  - An inverter is trained on 1B tokens.
  - The inverter is warm-started on 4 bullet rollouts, chosen by non-negative OMP to maximise FVE.
  - RL then rewards bullet FVE **with the AR frozen**.
- Agentic evals (CHIVE-style) are only preliminary.

### 1.2 Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (transformer-circuits, July 2026; arXiv 2607.15495) [READ full HTML via grep]
https://transformer-circuits.pub/2026/workspace/index.html

**Definition.** Five properties: verbal report, directed modulation, internal reasoning, flexible generalization, selectivity.

**J-lens**
- J_ℓ = E[∂h_final,t′/∂h_ℓ,t], averaged over source position t, all t′≥t, and about 1,000 prompts of 128 tokens.
- lens(h) = softmax(W_U · norm(J_ℓ h)).
- J-space = "sparse nonnegative combination of J-lens vectors" (k≈25).
- "Averaging across contexts distinguishes representations that are verbalizable—poised to be spoken about, should the occasion arise—from those that merely happen to be verbalized in one particular context."
- J-lens beats logit lens and tuned lens with as few as 10 prompts.
- The tuned lens "tends to 'skip ahead' to the output rather than surface those intermediates."

**Key numbers**
- J-space's excess variance explained "never exceed[s] 10%" ("activations are dominated by information outside the J-space").
- Workspace layers ≈ 38–92 on a 0–100 scale.
- Two-hop probe decomposition:
  - The J-space component (k=25) explains about 10–15% of probe variance.
  - Swapping it flips the answer on **61%** of trials, matching 60% for raw J-lens swaps.
  - Swapping the non-J remainder flips **28%** (n=90), "and this residual effect is itself routed through the J-space."
- J-space ablation (top-10 directions): multihop drops to about zero while pretraining top-1 match stays >95%.
- Shallow tasks (MMLU, SQuAD, sentiment, CoLA, odd-one-out) are "essentially unaffected even under heavy ablation."

**Selectivity: the most important section for this project**
- **Part of speech and tense (paired-question protocol).**
  - Next-word condition: the model produces an adjective ("unacceptable"), "yet neither adjective nor adj appear in the J-lens readouts."
  - Name-the-property condition: adjective-related tokens appear at 3 stimulus positions.
  - Tense behaves the same way.
  - "the property is represented and in use under both questions; what the question modulates is whether its label enters the J-space."
- **J-orthogonalized probes (appendix, "Directed modulation affects the J-space more than other representations").**
  - Four categories: coding language, verb tense, part of speech, capitalization; n=24 stimuli each.
  - The probe is a mean-difference probe with the top k=25 J-directions projected out.
  - An "imagine" instruction moves the J-lens by several SD and leaves the probe "within roughly one standard deviation."
  - A real stimulus moves the probe **3–6 SD**.
  - "a past-tense sentence drives the tense probe up six standard deviations, while past remains far down the lens ranking. The model can thus represent and use a property without that property's label entering the J-space."
  - **[INFERRED] This is a ready-made ground-truth instrument for non-workspace content.**
- **Line-width / character count** (Gurnee et al. linebreaks task, n=11 passages).
  - Under automatic linewrap, "number tokens are entirely absent from the lens," and a lens swap (40s→60s) leaves the wrap point unchanged.
  - Under explicit report, numbers appear at 20 positions and the swap moves the answer from 46 to 65.
  - Presence is "essentially zero under linewrap."
- **Language of passage** (n=8). The language name *is* in the J-lens in all four tasks, but the swap affects only explicit report and flexible inference, not continuation or anomaly detection. This is a "present in J-space but causally unused" category.
- The introduction lists non-workspace "low-level bookkeeping—the part of speech of the present word, or the length in characters of a line of text."

**Oracle Lens appendix**
- Built on Haiku 4.5, layer L67.
- "Explains on average 31% of the whitened activation variance."
- The reconstructor is trained in advance and frozen. Reconstruction is a linear combination of template vectors.
- Key quote: "we suspect that the oracle lens is less likely to produce confabulations than NLAs. On the other hand, it also achieves considerably lower reconstruction accuracy. **One interpretation of this is that the oracle lens extracts only those representations that are 'in the workspace,' while NLAs attempt to reconstruct the entire activation vector.**" This is the origin of the hypothesis under test.

**Limitations they state**
- Single-token vocabulary ("prompt injection" comes out as "prompt" and "injection"). The template lens is offered as a multi-token extension.
- "Beyond a bag of concepts": binding and roles are invisible.
- Some top tokens are uninterpretable.
- Early-layer workspace may be missed.
- "automatic, well-practiced computations proceed without engaging the J-space; a misaligned behavior that has been reinforced enough to become a fixed circuit might do likewise."

### 1.3 Anthropic, "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations" (Fraser-Taliente, Kantamneni, Ong, Marks et al., May 2026) [READ full HTML via grep]
https://transformer-circuits.pub/2026/nla/

**Method**
- The AV (verbalizer) and AR (reconstructor) are both copies of the target model.
- The activation is injected as a token embedding.
- Warm start: Claude-written summaries of truncated pretraining text, giving FVE ≈ 0.3–0.4.
- RL: GRPO with reward −log‖h−AR(z)‖², KL toward init, reaching FVE ≈ 0.6–0.8 ("roughly linear in log steps").
- Gemma-3-27B RL went from 0.38 to 0.71 FVE in 1.5 days on 2×8 H100.
- Trained at **two-thirds depth**. About 500 output tokens per activation.

**Evals**
- Five prediction tasks improve with training. Example: Classification goes 83%→86% on Opus 4.6.
- **"an LLM provided the full input as context would likely attain a high score on all of these tasks. Thus, these evaluations do not measure whether NLAs are uniquely useful."** No text-only baseline is reported.

**Steganography test**
- Meaning-preserving transforms (shuffle, paraphrase, French translation) cause small FVE drops.
- Coherence and summary rewrites cause large drops.

**Confabulation**
- "claims about themes of the text are more likely to be supported than those about specific details."
- Removing true claims hurts reconstruction more than removing false ones, so the AR is a "weak per-claim verifier."
- True claims recur across adjacent tokens more often.
- The rate of verifiable confabulation is "roughly flat over training."
- Exact percentages are interactive only. Zhang (below) quotes the Opus 4.6 NLA as plausible on **64% of theme claims, 28% of entity claims and 24% of detail claims**.
- These results "characterize confabulation only for claims about the input text, not for claims about model cognition."
- They propose two origins: the SFT warm start, and the reconstruction objective ("490" guessed for a weakly represented "491").

**Input inversion**
- "We do observe partial input inversion — NLA explanations sometimes quote sentences from the context verbatim — but this does not dominate."
- Full inversion is prevented by a bottleneck of under 500 tokens.

**Verbalizability (direct quotes)**
- "Some information may be 'unverbalizable.' ... encoded in a form the model cannot introspect on or express in language. For instance, a model with an implanted backdoor may not be able to verbalize its trigger even though the trigger is detectable with mechanistic techniques... The extent to which verbalizability limits NLAs is currently unclear."
- Future work: "**Characterizing what NLAs cannot verbalize.** We have speculated that some activation content may be unverbalizable, but this is currently untested."
- "NLAs may in principle be able to access information stored nonlinearly."
- Note: the NLA paper frames the AV as a model verbalizing activations. It does not discuss the J-space, which came later.

**Unverbalized evaluation awareness**
- Deployment noise floor: 2/50 (4%).
- NLA-measured awareness exceeds 5% on 10 of 16 evals, 7 of which have 0% verbalized awareness.
- Steering ±0.1 moves NLA-measured awareness in the predicted direction.
- They note: "asking the model a follow-up question... might also track credence."
- **[INFERRED]** "Unverbalized" here means *not said spontaneously*. It does not mean unverbalizable or outside the workspace. The J-space paper finds evaluation awareness (fake, fictional) *inside* the J-space.

**Single layer.** An NLA at 2/3 depth missed the reward-model-sycophancy root cause. One at about half depth found it.

### 1.4 Jakkli (aryaj), "Current activation oracles are hard to use" (LW, 2026-03-03, MATS 9.0, Nanda and Rajamanoharan) [READ]
https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use

Setup: Karvonen et al. AOs, Qwen3-32B at layer 32/64.

**Vagueness and hallucination**
- **49.4%** of responses were generic (77 probes over 33 rollouts).
- Hallucinations such as "BANANA" letters and "eggs laid by hens".

**Text inversion**
- About 600K of the AO's roughly 1M training examples are previous/next-word identification. "correct answers don't prove the AO is reading deeper signal."
- "It's an LLM": running the AO with *no* activations gave comparable or better answers on the Chinese-censorship prompts.

**Tasks where text inversion cannot help**
- *Number prediction*: the AO says "the number 10" / "value of 12" for every problem.
- *Sycophancy* (Scruples): AO AUC ≈ 0.55–0.60, versus 77% for a dense probe (evaluated on training data, so optimistic).
- *Missing information*: the AO identifies the right source 1/54 times. It gives the same classification in 53/54 cases when tokens are identical but model state differs (condition A vs C). It flags missing information 59% of the time when nothing is missing.

**Backtracking**
- About 10% correct and specific, about 5% after a text-inversion filter (100/2,300).
- Edit test: 8/15 updated correctly, but all 8 just restated a keyword from immediately before the probe.

**Positive signal**
- 4-option MCQ: 66% versus 47% for an LLM.
- Next/previous-token F1: 0.38 / 0.45 versus 0.30 / 0.20 for an LLM.
- Pirate steering is detected at coefficient 0.15, before it shows in the text.

### 1.5 Karvonen, "3 Tips to Improve Activation Oracle Results" (LW, 2026-09-28) [READ]
https://www.lesswrong.com/posts/XL3Rdq8mZEnqYym4B/3-tips-to-improve-activation-oracle-results

**Tip 1: use many tokens**
- Qwen3-8B backtracking with a single-token window is near chance.
- At 20 tokens it matches a full-text Qwen baseline. At 50 tokens it beats it.

**Tip 2: consensus sampling**
- 10 samples at temperature 1.
- Taboo task: 46.6% accuracy unfiltered. Requiring consensus ≥0.8 keeps 19.4% of examples at **94.3% precision**.

**Tip 3: use AUC instead of accuracy**
- Sycophancy: 0.50 accuracy but **0.83 AUC**.
- Accuracy swings about 20pp with prompt wording.
- Karvonen: AOs "frequently... obtain a similar AUC to a trained linear probe, which I interpret as the skyline."

**On NLAs**
- About 200 tokens per position, so 1,000 tokens of CoT means about 200K tokens of noisy analysis.
- Expects AOs to win on concepts spread over many tokens.
- Key quote: "**AOs, NLAs, and SAEs provided no uplift over a strong LLM that just read the transcript** when predicting model behavior. This could be because current LLMs don't do much computation that can't be inferred from the text."

### 1.6 HF model cards [SUMMARY via WebFetch; values look verbatim]

**`ceselder/qwen3.6-27b-nla-rl`** (https://huggingface.co/ceselder/qwen3.6-27b-nla-rl)
- EasyNLA. Layer **42** (⌊2·64/3⌋). d=5120. Injection token `㈜` (id 158983). Inject at layer 1, output `<explanation>`.
- Hybrid architecture: 48 gated-DeltaNet layers plus 16 full-attention layers.
- FVE: 53.0% at step 0, peak about 78% near step 800, 75.6% at checkpoint 300, 73.9% at step 600.
- GRPO: group size 8, temperature 1.0, k3 KL β=0.01, hinged length penalty of 0.01/token over 192.
- **Warmstart data: `ceselder/qwen3-8b-nla-L24-finefineweb-100k`**, with activations regenerated at L42.
  - That dataset card says explanations were written by **Claude Sonnet 4.6 from the input prefix text only, not from activations**.
  - 100K FineFineWeb docs across 67 domains. 994K rows (av_sft 247K, ar_sft 247K, rl 500K).
  - **[INFERRED]** This confirms the concern: the AV is initialised to describe the *text*.
- Note: nla-vs-lenses used adapter `iter_000400`.

**`agu18dec/olens_and_ar`** (https://huggingface.co/agu18dec/olens_and_ar)
- A matched pair of LoRAs: Oracle Lens (AO) plus Activation Reconstructor (AR, "the ruler").
- AO: 600 GRPO steps (β 0.02) from the bullet student. **Trained on layers 20–60, step 4** ("Layers outside 20–60 step 4 are off-contract"). Unit norm with α=16000. "concepts_raw" prompt.
- AO held-out joint FVE **0.487**.
- AR mean validation FVE 0.226, retrieval@1 0.94. Per layer: L20 .120, L28 .146, L36 .199, L44 .220, L52 .292, L60 .341.
- Warnings:
  - Capture activations with `disable_adapter()`. "Reading the lens's own modified activations instead of the model's is a silent, plausible-looking error."
  - The prompt, transform and α must match exactly.
- **[INFERRED]** OL FVE (whitened, bullet-template reconstruction) and NLA FVE (raw-space, learned AR) are **not comparable numbers**.
- nla-vs-lenses used `agu18dec/local-workspace` `s3d.ddp600.s0/iter_000600`, the WorkspaceBench default.

### 1.7 Highly relevant 2026 work found by search [READ unless marked]

**Viswanath, "Models are blind outside the J-space. NLAs aren't." (LW, 2026-07-08)** [READ] https://www.lesswrong.com/posts/LhDJdccLszLEAqgZ9/nlas-read-thoughts-beyond-the-j-space
- Setup: Llama-3.3-70B, Anthropic's open NLA at L53, plus a self-trained J-lens.
- Concept vectors: difference of means over "Tell me about {word}" for 24 concepts.
- De-risking:
  - The NLA names 24/24 injected concepts.
  - Gaussian vectors read as silent.
  - Mislabeling does not fool it.
  - Neutral carrier at strength 1.0: named 46% with injection, 0% without.
- J-fraction at r=200 is 6.1%, matching the paper's 6–7%. r=1076 gives 50% and r=4450 gives 90%, so there is no clean wall.
- **The pilot flipped as n grew.**
  - At r=200 with 6 concepts: P(yes) 0.025 for J versus 0.002 for non-J; the NLA read non-J at 0.83 and J at 0.44.
  - With **n=16 the pattern reversed**: the model's report tracked non-J.
  - Diagnosis: the r=200 "non-J" part still holds about 94% of the variance plus workspace content.
- P(yes) is unreliable: random in-J directions gave P(yes) 0.077–0.095, *higher* than real J components at 0.038. He switched to name-out-loud rates.
- **Result at r=2000**: the model names J 80% and non-J 0%; the NLA names both 100%. There is a dip beyond r=2000.
- Franken-vectors (6 pairs, strength 4): the model named a concept 3/12 times, all the J-side concept. The NLA named both 12/12.
- The non-J component raises the concept token's logprob by **+13.7 to +21.1 nats** while the model never names it.
- Caveats from the post:
  - The NLA confabulates context ("Batman franchise").
  - Sometimes it gives only "a specific emotion".
  - **On natural line counting the NLA never reads the exact character count that the J-lens shows.**
- **[INFERRED] Weaknesses for Neel's question:**
  - Synthetic, norm-matched injections are off-distribution.
  - The non-J part still carries the concept's *unembedding* direction (+21 nats), so "the NLA reads it" may be read-out of token identity, not "subconscious" content.
  - No Oracle Lens. Small n. One layer.

**Hong, "Fixing rewards for NLA to reduce confabulation" (LW, 2026-07-24)** [READ] https://www.lesswrong.com/posts/DFgt8fi3Wzwwe2Sib/fixing-rewards-for-nla-to-reduce-confabulation
- Anthropic's public checkpoints score FVE 0.712 on a held-out split.
- Their own Qwen3.6-27B run: FVE 0.185→0.413.
- **About 90% of specific claims (names, numbers, quotes) are unsupported.**
- 10 LLM judges preferred post-RL over pre-RL only 54.5% of the time [45.2, 63.5].
- Entity-swap ΔFVE −0.219 versus abstract-swap −0.072. On Anthropic's checkpoint: −0.394 versus −0.103. So the "AR reads the (false) named entities."
- A regex fact-check reward raised the fact score from 0.124 to 0.417 with FVE flat (0.4126→0.4194). Judges preferred it 83.6%.
- **J-space split of reconstruction:** inside-J is "nearly saturated from the start," outside-J "almost tripled." "About 95% of training gains live outside the J-space." The split is unchanged by the fact-check reward.
- Hong cites, without a link, a LW post showing that "a LLM [that] never sees the activation at all, could invent a description from the text and reconstructs about as well as the real AV." **I could not locate that post.**

**Zhang, "How robust are natural language autoencoders to initialization?" (LW, 2026-07-10, MATS team shard)** [READ] https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to
- Qwen2.5-7B, 20K docs.
- An **implausible-initialized NLA reaches FVE 0.68, versus 0.70 for a plausible-initialized one, while emitting 99.3% implausible claims.**
- RL lowers the plausibility of the plausible-initialized NLA from 21% to 7.6%.
- Appended addenda ("Carthage must be destroyed") are stripped in fewer than 100 RL steps.
- Claude quotes the correct final token even when told to confabulate. An NLA trained on only the last paragraph reaches FVE 0.67.
- "Earlier work found... the final paragraph about the last token is by far the most important."

**jaredcobrin/nla-sae-grounding (GitHub)** [SUMMARY] https://github.com/jaredcobrin/nla-sae-grounding
- Gemma-3-12B-IT at L32, 200 activations.
- The final paragraph alone keeps 84% of FVE (0.582 / 0.691) and 84% of recovered SAE latents.
- 43% of labelled claims are grounded.

**LittleMeHere/nla-vs-lenses (GitHub, 2026-10-04, MATS 12 mini project)** [READ STATUS.md] https://github.com/LittleMeHere/nla-vs-lenses
- Setup: Qwen3.6-27B, the same NLA (iter_000400), OL `agu18dec/local-workspace` iter_000600, Neuronpedia J-lens.
- Multihop (n=100, L42): J 39, OL 80, NLA 75. Chance 0.7 / 1.7 / 1.2.
- At L28 (n=50): J 1, OL 26, NLA 27.
- Deleting the bridge-neighbourhood directions (about 31): J 15→0, OL 33→20, NLA 34→23. "About two-thirds survive", which they read as "mostly re-solving."
- Swap: own concept kept 20 times, planted concept read 6–7 times. The planted concept is weak (J sees it 14/50 even at 16× scale).
- Chain (arithmetic) at L56: OL **36/60** (chance 12), and 31 of those hits do not restate the start number. NLA 13 (chance 9). J 25 (chance 7).
- Chain at L42: no reader is clearly above chance (J 6/4, OL 27/21, NLA 9/12).
- Brew: role-correct mentions are only 2–5/50, and they go to about zero under deletion.
- **NLA text on multihop:** 1,696 concepts.
  - 12% hidden step or answer, 15% also in J top-10, 22% prompt words, 22% format, 27% other.
  - Of "other": 22% contradicted by the prompt, 27% loose associations.
  - **66/98 readouts contain at least one contradicted claim.**
- "On no task does the NLA find a hidden step the oracle lens misses."
- Untested: content outside J-space.

**Bhatia, Blank, Nanda, "Towards surfacing model algorithms with meta-tokens in the J-space" (LW/AF, 2026-07-20)** [SUMMARY] https://www.alignmentforum.org/posts/6ek6n7yZ5DzfarJHy/towards-surfacing-model-algorithms-with-meta-tokens-in-the-j
- On Qwen3.6-27B, J-lens surfaces process tokens: 什么意思 for ambiguity, "gcd" in LCM problems, 大概率 before hedging.
- Steering these changes behaviour.

**Prabhu, "Can You Hide From a Natural Language Autoencoder?" (LW, 2026-06-24)** [SUMMARY] https://www.greaterwrong.com/posts/3oRQxpxn4LiWKwbeQ/can-you-hide-from-a-natural-language-autoencoder
- On Qwen2.5-7B, a prefix-tuned activation flips the AV's explanation while behaviour is preserved.
- Taboo-word models: the auditor wins.

**Kamesh R, "Exploring Generalization in NLA's" (LW, 2026-06-25)** [READ] https://www.lesswrong.com/posts/tkiSQBuA8yj2tHNdv/exploring-generalization-in-nla-s
- A Qwen2.5-7B L20 AV transfers to L10–25 above random and shuffled baselines.
- **[INFERRED]** Applying an NLA off its training layer is semi-viable but degraded.

**Bauer, De Schamphelaere, Karvonen, Luick, Nanda, "Building Better Activation Oracles" (ICML 2026; arXiv 2606.02609)** [SUMMARY] https://www.lesswrong.com/posts/heXwuDRfbQQgB5JLP/building-better-activation-oracles
- On-policy data and multi-layer input. Releases **AObench**.
- Capability gains are "marginal". Multi-layer AOs give no downstream uplift.

**Anthropic CHIVE (Aug 2026)** [SUMMARY] https://alignment.anthropic.com/2026/chive/ · arXiv 2608.16747
- AO, NLA and SAE tools give **no uplift** over transcript-only predictors.

**Bersia and Gaintseva, "When Activation Oracles Learn Not to Read" (arXiv 2607.23379)** [SUMMARY] https://arxiv.org/abs/2607.23379
- AOs fine-tuned on concept-hiding models develop concept-specific blind spots, even though the concept stays decodable.
- "behavioral leakage, representation-level decodability, and AO-verbalizability can come apart."

**Ferrara, "Open-Weight Masked Introspection" (arXiv 2608.20569, Aug 2026)** [SUMMARY] https://arxiv.org/abs/2608.20569
- 78K measurements over 8 models: models **cannot report** whether their computation was intervened on (AUROC ≈ 0.5007).
- **Linear probes detect the intervention at 75–95.8%.**
- One model's confidence (not its words) reaches AUROC 0.647.

---

## 2. Novelty check, in detail

| Claim to test | Status | Evidence |
|---|---|---|
| NLA reconstruction gains are mostly outside the J-space | **Done (Qwen3.6-27B)** | Hong: about 95% of RL FVE gains are outside J. Geometry only. |
| NLA can name *injected* concept components the model cannot report (non-J) | **Done (Llama-3.3-70B, NLA only)** | Viswanath r=2000: model 80% / 0% (J / non-J); NLA 100% / 100%. No OL, no Qwen. |
| NLA beats J-lens on WorkspaceBench intermediates because it reads more workspace content | **Partly done; mostly explained by re-solving** | nla-vs-lenses multihop: J 39, OL 80, NLA 75, but about 2/3 survive deletion. |
| Oracle Lens genuinely reads numeric intermediates the NLA misses | **Done (pilot)** | nla-vs-lenses chain at L56: OL 36/60 versus NLA 13/60. WorkspaceBench chain: OL .25, J .22, NLA .03. |
| NLA reads *natural* non-workspace content (POS, tense, linewrap counts) that J-lens and OL miss | **OPEN** | Only Viswanath's preliminary line-counting note, and it is *negative*: the NLA does not read the exact count. |
| Oracle Lens is blind to non-J content ("OL reads only workspace") | **OPEN (hypothesis only)** | The J-space paper's appendix offers it as "one interpretation". Nobody has tested it. |
| Fraction of NLA-only claims that are prompt restatement vs confabulation vs verified non-workspace | **OPEN (first pass on multihop only)** | nla-vs-lenses buckets show 22% prompt words, 22% format, 22% of "other" contradicted. No probe verification. |
| What NLAs cannot verbalize | **OPEN** | The NLA paper explicitly says "currently untested". |
| Text-only baseline (an LLM describing the prefix with no activation) for NLA FVE and for WorkspaceBench content | **Partly done** | Zhang: implausible text still gives FVE 0.68. Hong cites an unlocated post. WorkspaceBench's prompt-only baseline is strong. Nobody has a per-claim text-only control on Qwen3.6-27B. |

**Bottom line.** The *direct* question has been posed in public by a concurrent MATS 12 pilot, which says it is unanswered. That pilot's next step (#2, split injection into all three readers) is one obvious design. The differentiated contribution is **natural-activation, probe-grounded categories of non-workspace content**, plus a **text-only control**, run on all three readers at L42.

---

## 3. Candidate categories of "in the residual but not verbalizable / not in the workspace"

Each entry gives the literature support, how to build test items, and what counts as ground truth.

**1. Part-of-speech of the upcoming word, and verb tense, under automatic continuation**
- Support: J-space paper, paired-question protocol plus J-orthogonalized probes. The probe moves 3–6 SD on real stimuli while the label stays out of the J-lens. Earlier probing work: Hewitt and Manning 2019, structural probe (https://aclanthology.org/N19-1419/); Tenney et al. 2019, "BERT rediscovers the classical NLP pipeline" (https://arxiv.org/abs/1905.05950).
- Build:
  - Passages that force an adjective, or a past-tense verb, next. Use the next-word instruction, not a name-the-property question.
  - Read J, OL and NLA at stimulus tokens at L42.
  - Ground truth: a mean-difference probe with the top-k J-directions projected out, trained on real positive vs negative stimuli.
  - Score whether each reader states "adjective" / "past tense" / "verb".
  - Control: the name-the-property condition, where the label *should* enter the J-space.
  - Text-only control: Claude reads the prefix and is asked what the model is tracking.
- **Caveat [INFERRED]:** a prose reader can trivially infer the POS from the decoded prompt. You need the text-only baseline, and also cases where the property is in the activation but hard to infer from the prefix (for example, conflicting cues).

**2. Running character / column count during automatic linewrap**
- Support: J-space paper (numbers absent from the lens under linewrap, and the swap has no effect); Gurnee et al. 2025, "When Models Manipulate Manifolds" (https://transformer-circuits.pub/2025/linebreaks/index.html); Viswanath's preliminary note that the NLA does not read the exact count.
- Build:
  - Fixed-width passages with lines of 30–70 characters and the instruction "continue, preserving wrapping."
  - Read at the newline or the final token.
  - Ground truth: the exact column number, plus a linear or manifold probe for character count.
  - Score whether a reader states a number within ±2.
  - Compare with the explicit-report condition.
- A strong test, because the content is numeric and specific, so prompt restatement can't easily fake it. Prose readers would have to count characters in the text they decode.

**3. Injected concept components outside the J-space**
- Support: Viswanath (above); Lindsey, "Emergent Introspective Awareness in Large Language Models" (2025, https://transformer-circuits.pub/2025/introspection/index.html; arXiv 2601.01828); Ferrara OWMI (probes 75–95.8% vs chance self-report).
- Build:
  - Difference-of-means concept vectors on Qwen3.6-27B.
  - Split at J-rank r, with a rank sweep (note the heavy tail: r=200 is too small).
  - Norm-match and inject at L42 into a neutral carrier.
  - Readers: model self-report (name out loud), J-lens, OL, NLA.
  - Controls: no injection, random in-J direction, random out-of-J direction, Franken-vectors.
  - Ground truth: by construction.
- **Also strip the concept-token unembedding direction from the non-J part** to rule out "the reader reads the token identity."

**4. Language of the passage during continuation**
- Support: J-space paper.
- This is a control category, not a non-workspace one: the content is *in* the J-space but causally unused. Use it to separate "present" from "used".

**5. String probability / grammaticality knowledge**
- Support: Song, Hu and Mahowald 2025, "Language Models Fail to Introspect About Their Knowledge of Language" (COLM 2025, https://arxiv.org/abs/2503.07513). Metalinguistic self-reports don't track the model's own string probabilities any better than another model's do.
- Build:
  - Minimal pairs (BLiMP-style).
  - Ground truth: the model's log-prob difference.
  - Ask whether readers' readouts encode the *degree* of surprise.
- Harder to score. Probably out of scope for 2 weeks.

**6. Hidden factual knowledge and error signals**
- Support:
  - Orgad et al. 2024, "LLMs Know More Than They Show" (https://arxiv.org/abs/2410.02707): truthfulness signals concentrate in exact-answer tokens, and probes predict errors.
  - Gekhman et al. 2025, "Inside-Out: Hidden Factual Knowledge in LLMs" (https://arxiv.org/abs/2503.15299).
  - Burns et al. 2022, CCS (https://arxiv.org/abs/2212.03827).
  - Azaria and Mitchell 2023 (https://arxiv.org/abs/2304.13734).
  - Marks and Tegmark 2023, "Geometry of Truth" (https://arxiv.org/abs/2310.06824).
  - Kadavath et al. 2022 (https://arxiv.org/abs/2207.05221).
- Build:
  - Items the model answers wrongly while a probe says the correct answer is represented, or where a probe predicts the error.
  - Check whether readers surface "I'm likely wrong" or the right answer.
- Note: error awareness may be verbalizable on request, so it is not cleanly non-workspace.

**7. Internal algorithm differs from the self-description**
- Support: Lindsey et al. 2025, "On the Biology of a Large Language Model", mental-math section (https://transformer-circuits.pub/2025/attribution-graphs/biology.html). The model describes standard carry addition while internally using parallel approximate-magnitude and ones-digit paths.
- Build: hard to ground-truth without circuits on Qwen. Low priority.

**8. Sycophancy / user-hint influence not mentioned in CoT**
- Support: Turpin et al. 2023 (https://arxiv.org/abs/2305.04388); Jakkli (dense probe 77% train-AUC, AO 0.55–0.60); Karvonen (AO AUC 0.83).
- Build: Scruples items with and without a hint.
- But it is likely in the workspace (Anthropic sees "suspicion"-type concepts in the J-space). Better used as a safety-relevant demo than as a non-workspace test.

**9. Positional / bookkeeping information (token index, distance to the delimiter)**
- Support: the J-space paper's "low-level bookkeeping"; Haviv et al. 2022, "Transformer LMs without positional encodings still learn positional information" (https://arxiv.org/abs/2203.16634); Gurnee and Tegmark 2023, space and time probes (https://arxiv.org/abs/2310.02207).
- Build:
  - Ground truth: a token-index probe at L42.
  - Ask whether readers state the position or the line number.
  - The text-only control is crucial, because a prose reader can count if it reconstructs the prompt.

**10. Backdoor triggers / implanted objectives**
- Support: the NLA paper's speculation; Hubinger et al. 2024, "Sleeper Agents" (https://arxiv.org/abs/2401.05566); MacDiarmid et al. 2024, "Simple probes can catch sleeper agents" (https://www.anthropic.com/research/probes-catch-sleeper-agents).
- Out of scope for 2 weeks on a 27B model unless an organism already exists.

**11. Subtle steering / interventions**
- Support: AO detects pirate steering at 0.15 (Jakkli); Ferrara OWMI (the model can't report an intervention, probes can).
- Build: low-coefficient steering vectors whose J-component has been removed.

---

## 4. Known failure modes, by method

**NLA**
- **Hallucination.**
  - WorkspaceBench: about 0.96 of NLA-RL readouts contain at least one wrong claim (about 0.97 for NLA-SFT), versus about 0.3 for OL and J.
  - Hong: about 90% of specific claims are unsupported.
  - Zhang: 7.6% plausible at the end of RL.
  - nla-vs-lenses: 66/98 readouts contain a prompt-contradicted claim.
  - Anthropic: theme claims plausible 64%, entities 28%, details 24% (as quoted by Zhang).
- **Confabulated specifics carry the reconstruction.** Hong: swapping entities costs −0.219 FVE, versus −0.072 for swapping abstract content.
- **Text-describing initialization.** The warmstart is Sonnet 4.6 descriptions of the prefix text only (dataset card). Zhang: implausible initialization still reaches FVE 0.68.
- **Input re-description / re-solving.**
  - nla-vs-lenses: 22% of concepts are prompt words and 22% describe format.
  - Readouts at L16 restate the question and answer it.
  - Anthropic: "partial input inversion."
- **The final paragraph dominates** (the last-token description): Zhang, and the cobrin repo (84%).
- **Single layer** (NLA paper).
- **Fooled by adversarial activations** (Prabhu).
- **Off-layer degradation** (Kamesh).
- **Cost:** about 250–500 tokens per activation.

**Oracle Lens**
- Lower reconstruction than the NLA. Haiku: 31% whitened FVE. Qwen: AO joint FVE 0.487, AR 0.226.
- Still hallucinates: WorkspaceBench about 0.34 (RL) and 0.28 (SFT).
- **Prompt echoing / text inversion.**
  - It is generally a continuation predictor at most positions: "at most positions it generates predictions of upcoming text" (J-space appendix). Running commentary appears only at delimiters.
  - nla-vs-lenses: OL names multihop bridges at L28, where the J-lens shows 1/50, and about 2/3 of mentions survive deletion, which means re-solving.
  - The broader AO literature (Jakkli): the no-activation AO matches the AO with activations; edit tests just restate nearby keywords.
- Setup gotchas: off-contract layers, the `disable_adapter()` capture requirement, and sensitivity to α and prompt.

**J-lens**
- **Single-token vocabulary.** "prompt injection" splits into "prompt" and "injection"; "black" stands in for blackmail.
- **Bag of words:** no order or binding (role-bound .13, relational multihop .01).
- Uninterpretable top tokens at some positions.
- Readable only from about the first third of the model onward.
- On Qwen chain intermediates it shows numbers mainly at L56 (nla-vs-lenses).
- Misses automatic computation **by design** (selectivity). That is the property under test, not a bug.
- R-lens and tuned lens: the tuned lens "skips ahead" to the output.

**General confounds for all prose readers**
- The reader is an LLM and can re-solve the question (Jakkli; nla-vs-lenses).
- Judges with weak models let false positives through (WorkspaceBench).
- Accuracy depends on the reader's default answer, so prefer AUC (Karvonen).
- Readers can share knowledge with the subject model (Jakkli, on Chinese censorship).

---

## 5. Highest-value open questions for a 2-week sprint (ranked)

**1. Natural non-workspace probe test at L42 (most novel).**
- Use the J-space paper's four probe categories (POS, tense, capitalization, code language) plus linewrap character count.
- Build J-orthogonalized probes on Qwen3.6-27B to confirm the property is present and outside J.
- Under the *automatic* condition, measure how often J, OL and NLA state the property.
- Compare against (a) the explicit-report condition and (b) a text-only Claude baseline given the prefix.
- **Prediction to pre-register:** if NLAs read non-workspace content, NLA > text-only on automatic-condition items while J ≈ 0. If the NLA only re-describes text, NLA ≈ text-only.

**2. Split-injection with three readers on Qwen3.6-27B.**
- Replicate Viswanath with J, OL, NLA and model self-report at L42, with a rank sweep.
- Remove the concept's unembedding direction from the non-J part.
- Directly tests "OL reads only workspace" (the J-space appendix's hypothesis) against "NLA reads beyond it."
- It is the cleanest ground truth, but nla-vs-lenses lists it as their next step, so it is lower novelty. Consider coordinating with them.

**3. Activation-level decomposition of NLA readouts.**
- Give the NLA (a) h with the top-r J-space directions removed, and (b) h projected onto the J-space only.
- Classify each claim as one of: survives J-removal, survives J-only, neither.
- Cross with a text-only baseline and a judge for contradiction.
- Answers what the NLA's unique claims are made of (prompt restatement, confabulation, verified non-J content) using Hong's geometric finding as the starting point.

**4. Matched-precision comparison.**
- Apply consensus or recurrence filtering to the NLA: 10 samples, plus Anthropic's adjacent-token recurrence heuristic.
- Compare against OL at matched hallucination rates on the WorkspaceBench hallucination eval.
- Asks whether the "NLA captures more" claim survives once precision is matched.

**5. Re-solving vs reading as a function of prompt recoverability.**
- nla-vs-lenses' untested explanation: a reader re-solves when it can rebuild the prompt.
- Measure per-item prompt recovery from each readout and regress hidden-step naming on it.
- CPU-only, and reuses their raw files.
- Cheap, but also on their list.

---

## 6. Sources not located or not verified

- The LW post Hong cites ("an LLM that never sees the activation... reconstructs about as well as the real AV"): **not found.**
- The "Li et al." privileged-information critique cited by Viswanath: not located.
- Exact WorkspaceBench numbers: get them from the repo, not from my chart readings.
- The NLA paper's per-category confabulation percentages are interactive figures. The 64 / 28 / 24% figures come from Zhang's quotation.
