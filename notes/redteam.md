# Red-team: "Is the extra stuff NLAs capture really non-workspace info?"

Mentor-mode review of the in-workspace / out-of-workspace set proposal. Short version: **the current design cannot tell you anything, because both hypotheses predict the same thing.** If "out-of-workspace" items are drawn from probe-decodable features of natural text, then an NLA that just re-describes the input will also stay flat across the two sets. Flat NLA plus dropping J-lens/Oracle is what you'd see if the NLA reads latents. It's also what you'd see if the NLA is a good text summarizer and the lenses are bad at describing surface properties. You need items where **the latent and the text come apart**, and you need to define "out-of-workspace" **without using any of the tools being compared**.

---

## 1. Objections, ranked

### O1 (fatal). The design does not discriminate the two hypotheses
- H_latent: "the NLA reads information that is in the residual but not in the workspace."
- H_redescribe: "the NLA reconstructs/describes the input text, and the residual at layer 42 contains a lot of information about the input text."
- Probe-decodable features from natural prompts (sentiment, topic, syntax, language, register, "is inside a quote", etc.) are almost all **functions of the local text**. Under H_redescribe the NLA also stays flat on them. So the headline prediction cannot falsify H_redescribe.
- The "input-inferability control" as written ("a decoder that rereads the prompt proves nothing") does not fix this. The NLA's AV only sees the activation, but **the activation is mostly an encoding of the recent text**, so "reading the activation" and "rereading the prompt" are close to the same operation. The control has to be at the *item* level (the ground truth must not be computable from the text), not at the *tool* level.

### O2 (fatal). "Probe-decodable but not self-reportable" is not the same as "outside the workspace"
Self-report failure has at least five causes. Only one of them is "not in the workspace":
1. **Elicitation failure**: wrong prompt, wrong framing, or the model doesn't know the question refers to its own internal state. ("What concept are you thinking about?" is weird to a chat model.)
2. **Position/context mismatch**: you probe the activation at token t in context C. Self-report happens at a *later* position in context C + question. The question changes the residual. The info may be present at t and simply not carried forward to the answer position. That's a routing failure, not a workspace fact.
3. **Lexicalization failure**: the model has the info but has no words for it (e.g. "this is the 3rd sub-token of a word", "residual norm is high"). A verbalizer shouldn't be expected to output that either.
4. **Trained suppression**: RLHF or refusals (e.g. "I don't have internal states").
5. **Actually absent from whatever circuit produces reports**: this is the thing you care about.

Minimum fix: define self-report as failure under **several elicitations**: free-form, K-way forced choice in text, **logit-based 2AFC with no generation**, few-shot with demos of correct introspective answers, and question appended *at* position t (not after a long gap). Require test-retest stability. Otherwise your out-of-workspace set is mostly "items where my prompt was bad," and regression to the mean will shrink every gap when you re-test.

### O3 (severe). Many probe-decodable features are things no verbalizer *should* output
A linear probe at 0.9 AUROC for "token index > 200" or "previous token was a comma" proves the info is linearly present. It does not mean it's semantically salient. A good verbalizer that prioritizes salient content *should* omit it. So the out-of-workspace set confounds "unreportable" with "low-salience/low-variance". Two consequences:
- Lens drops on out-of-workspace items may just be salience ranking, not access.
- NLA "success" on such items, if any, is likely driven by FVE: the reconstructor is rewarded for capturing **high-variance** directions (token identity, position, attention-sink and massive-activation dims), not for capturing *latent* content.

Fix: score with **queries, not free generation**. For every item, all tools get the same K candidate answers and the judge (or a logprob scorer) picks one (see O5). That turns "did the tool happen to mention X" into "does the tool's output contain information that discriminates X from its foils".

### O4 (severe). Unmatched difficulty and selection on the outcome
Out-of-workspace items are selected because one verbalization channel failed. They will systematically be more abstract, lower-norm, lower-SNR, and more OOD for every verbalizer. "All tools drop" is uninterpretable, and "NLA flat" may just be floor/ceiling. Fix: **match the sets on probe AUROC and on direction norm/variance explained**. Report detection as a function of a dose variable you control (see Design A), not as a two-bin comparison.

### O5 (severe). Hallucination base rate and judge leniency
The NLA is known to hallucinate heavily, and its output is long prose. A lenient judge reading a 10-bullet paragraph will find *something* matching X at a high rate. The current "lucky-guess / prompt-only floors" are per-family. You need **per-item null distributions**:
- Run every tool on **matched null activations**: same text, no injection / other item's latent / shuffled position. Report **AUROC or d′ between true and null**, not hit rate.
- K-way forced choice with foils of the same type ("which of {ocean, volcano, violin, tax law} is present?"): chance = 1/K, and it's symmetric across prose vs token outputs. That kills weakness #3 (format asymmetry) at the same time.
- Length-control: truncate NLA output to N claims, or have the judge score atomic claims for precision **and** recall.

### O6 (serious). NLA training distribution is a confound in both directions
- Warmstart = LLM explanations of web text → the AV's prior is "describe the topic/genre/register of this text". Items whose ground truth is a web-text-describable property get a free boost. Out-of-workspace items built from natural text are disproportionately this kind of property → bias **toward** your hypothesis.
- Injected / patched / off-manifold activations are OOD for the AV → bias **against** it. An NLA failure on steered activations could just be distribution shift.
- GRPO on reconstruction FVE (~76%) means the AV is rewarded for **whatever text lets the AR rebuild the vector**. That's not the same as being faithful about content. Steganographic or idiosyncratic phrasing can carry FVE. 76% FVE says nothing about whether the 24% it misses is exactly the "latent" part.
Fix: always include an in-distribution control (natural activations from contexts that genuinely contain the concept) next to every intervention condition. Also inject at natural norms (fraction of median residual norm at that layer).

### O7 (serious). The premise itself is unverified
"NLAs capture more than J-lens/Oracle" is presumably measured on WorkspaceBench with a prose judge. Before explaining *why* NLAs see more, check that they *do* under fair scoring: forced choice, null-controlled, same layer (42) for all tools, and an output-length control. I'd bet 40% that a large part of the gap is verbosity × judge leniency. That would also be a reportable result.

### O8 (moderate). Circularity, beyond the J-lens point you already listed
- Oracle Lens is a LoRA on the same model, trained to verbalize. Its ceiling is plausibly *the model's own verbalization pathway*. So "Oracle fails where self-report fails" is partly by construction, and it doesn't tell you about the workspace. Same caveat if the NLA AV is a fine-tune of Qwen.
- Fix: define the out-of-workspace region by **an external dose variable** (injection strength below the self-report threshold, measured with the 5-way elicitation in O2), not by any lens.

### O9 (moderate). Layer and position mismatch
NLA is layer 42 only. Compare J-lens/Oracle **at layer 42 on the same token** as the primary comparison; other layers are secondary. Also be explicit about read position vs inject/compute position. "Info present at t" ≠ "info present at t+5".

### O10 (moderate). Sprint scope
A 27-family benchmark, two hand-built item sets, three tools, a judge, and a self-report pipeline do not fit in two weeks solo, at least not done well. Pick one family where ground truth is known by construction and go deep.

---

## 2. Alternative / sharpened designs

### Design A: Injection dose-response ("the threshold gap")  [RECOMMENDED CORE]
- **Question:** As a concept vector injected mid-sequence weakens, does the NLA keep detecting it after the model's own self-report (and Oracle/J-lens) has dropped to chance?
- **Key experiment:** For ~30 concepts, build vectors (mean diff at layer L_inj ∈ {20, 30}; also a few SAE/probe directions that aren't "topics"). Inject at one position (or a short span) in neutral filler text, at strengths α spanning 0 → well above the self-report threshold, scaled to median residual norm. Read at **layer 42** at the injected position and at +k later positions where the text is neutral. Run five channels: linear probe (trained on natural data, not on injections), self-report (Lindsey-style "do you detect an injected thought / which of K?", logit 2AFC), J-lens, Oracle, NLA. Score each with K-way forced choice against null (α=0 and random-direction equal-norm) activations. Plot AUROC vs α for each channel; extract α50 per channel.
- **Ground truth:** known by construction and **not inferable from text** (text is identical across α).
- **Decisive predictions:**
  - H_latent: α50(NLA) < α50(self-report) ≈ α50(Oracle). There's an α band where NLA ≫ chance and self-report ≈ chance. That band is your operational "out-of-workspace" region, and it's defined by dose, not by a lens.
  - H_redescribe / no special access: α50(NLA) ≥ α50(self-report). Or NLA only detects concepts once they bleed into predicted next tokens, so J-lens detects them at the same α.
  - Also informative: probe α50 ≪ every verbalizer → info is present but no verbalizer reaches it (a clean "probe-only" regime).
- **Day-1 de-risk:** 5 concepts × 6 strengths × 20 contexts. Check (i) that self-report has a measurable threshold at all, (ii) that NLA false-positive rate at α=0 is < ~30% under forced choice, and (iii) that the NLA isn't just garbage on injected activations at natural strength (in-distribution control: natural activations of texts about the concept).
- **Cost:** ~1 H100 for the model plus one for the NLA. ~30 concepts × 8 α × 50 contexts × 5 channels ≈ 60k generations ≈ a few GPU-hours with vLLM/batched hooks. 3–4 days of work including plots.
- **Why Neel cares:** ground truth is known, causal, and not text-inferable. It ties directly to the introspection literature (Lindsey et al. 2025). It gives a single killer plot (five dose-response curves). It's falsifiable on day 2.

### Design B: Same text, different latent (2×2 crossed)
- **Question:** Does NLA output track the latent or the text when the two are decoupled?
- **Key experiment:** Construct a 2×2: {text same, text different} × {latent same, latent different}.
  - Text same / latent different: identical local window (last ~200 tokens), but a hidden variable set **far earlier** (e.g., "Everything below is from Alice's diary; Alice lives in Lisbon" vs "…Bob, Oslo", then 1–2k identical tokens). Alternatively, **activation-patch** the hidden variable's representation at its source position from a counterfactual run.
  - Text different / latent same: paraphrase the local window while keeping the hidden variable fixed.
  Read at layer 42 at a suffix token. Score whether NLA/lens outputs predict the latent (forced choice) and whether the outputs change with text when the latent is fixed.
- **Validation:** probe at the suffix decodes the hidden variable. A prompt-only LLM given **only the local window** is at chance. Patching the source changes downstream behavior (model answers "Where does the narrator live?" differently).
- **Decisive predictions:** H_latent: NLA sensitive to latent (Δ forced-choice accuracy ≫ 0) in the text-same cell, insensitive in the text-different/latent-same cell. H_redescribe: the reverse. Bonus: compare against self-report at the suffix ("where does the narrator live?"). The model will often *answer correctly*, so this family is mostly **in-workspace nonlocal** info, which is a useful positive control for Design A.
- **Day-1 de-risk:** check that the hidden variable is probe-decodable at layer 42 at 500/1k/2k tokens of distance. If the probe dies by 500 tokens, shorten.
- **Cost:** long contexts make each forward pass expensive but still cheap in absolute terms. 2–3 days.
- **Why Neel cares:** this is the cleanest "is it reading the latent or the text" test. It is exactly the illusion he warns about, and a simple factorial design removes it.

### Design C: Error-tracking / planted intermediates in no-CoT computation
- **Question:** When the model's internal intermediate is *wrong* (or planted by patching), does the NLA report the **model's** intermediate or the **correct** one a text-reasoner would infer?
- **Key experiment:** Two-hop factual prompts ("The spouse of the performer of <song> is") and 2-step arithmetic ("(17+25)*3 ="). (a) Natural errors: items where the model's final answer reveals a wrong bridge or a wrong intermediate. (b) Planted: patch the bridge-entity representation at the bridge position (mid layers) from another prompt so the final answer flips to the planted chain. Text is unchanged. Read NLA/lenses at the final token, layer 42.
- **Decisive prediction:** a faithful latent reader reports the **planted/erroneous** intermediate. A text re-describer (or any decoder that's secretly doing the task from the input) reports the **correct** one. This is the sharpest possible dissociation because a smart prompt-reader is *anti-correlated* with the right answer.
- **Day-1 de-risk:** find ≥100 two-hop items where (i) Qwen3.6-27B gets it right without CoT, (ii) a probe or J-lens finds the bridge at some layer, and (iii) patching flips the answer ≥50% of the time. Two-hop latent reasoning is notoriously flaky (see "hopping too late"). Arithmetic intermediates are the fallback because they're more reliable.
- **Cost:** patching sweeps are cheap. 3 days.
- **Why Neel cares:** it's a direct test of faithfulness vs plausibility, which is the central failure mode of all LLM-as-explainer tools, with a ground truth the explainer can't fake.

### Design D: Premise audit, "how much of the NLA advantage is a text-only baseline?"  [PIVOT TARGET]
- **Question:** On WorkspaceBench, how much of NLA's extra coverage is reproduced by an LLM that sees only the preceding N tokens of text and writes an NLA-style explanation?
- **Key experiment:** Build "TextNLA": Qwen (or a frontier model) given the last N∈{8,32,128} tokens, prompted with the NLA warmstart style. Score NLA, TextNLA, J-lens, and Oracle under forced choice + null + length control. Decompose NLA outputs into atomic claims. Label each as text-inferable or not (TextNLA agrees / disagrees) and measure the precision of the non-inferable claims.
- **Decisive predictions:** if TextNLA(N=32) recovers ≥80% of the NLA–Oracle gap, the "extra" is input re-description. If NLA's non-inferable claims have precision well above the null, there's a real latent channel worth chasing with A/C.
- **Day-1 de-risk:** run on 3 families. Takes hours.
- **Cost:** < 1 GPU-day. 2–3 days including writeup.
- **Why Neel cares:** a simple baseline that might deflate a flashy method. He explicitly values this, and it's a guaranteed-useful result either way.

---

## 3. Candidate "out-of-workspace" item types (ground truth not text-inferable)

Validate **every** type with the same four gates:
- (G1) Probe decodes it at layer 42 at the read position (held-out AUROC ≥ 0.8).
- (G2) A prompt-only LLM given the local window is at chance. Also give it the full context where relevant.
- (G3) Self-report is at chance under all five elicitations (O2), stably across two re-runs.
- (G4) The latent is **causally used**: ablating or patching it changes some downstream behavior. Otherwise it's an epiphenomenal feature that nobody should expect a verbalizer to report.

| Item type | Construction | Not text-inferable because | Specific validation / caveats |
|---|---|---|---|
| **Sub-threshold injected concepts** | Concept vector at α below the self-report threshold, layer 20–30, mid-sequence | Text identical across α | G4: does sub-threshold injection shift next-token logits toward the concept? (it should, weakly). Watch for OOD artifacts; in-distribution control required. **Best item type.** |
| **Distant-context hidden variable** | Variable set 1–2k tokens earlier, identical local window | Local window identical | Mostly *in*-workspace (the model can answer). Use as positive control. Becomes out-of-workspace only when combined with a weak signal (variable mentioned once, obliquely). |
| **Non-canonical tokenization** | Same string, forced different token segmentation (e.g. "un|happi|ness" vs canonical) | Surface string identical | Model is famously bad at reporting its own tokenization (G3 likely passes). G4: does segmentation change outputs? Often yes (spelling tasks). Caveat: a verbalizer arguably *shouldn't* report this (O3). Use as a "probe-only" calibration item. |
| **Role / provenance** | Identical text placed in system vs user vs tool-output vs assistant turn; read at a token inside the text | Text inside the span identical | Probe for role at layer 42 is easy. Model can sometimes self-report. Check G3 carefully. Interesting safety angle (prompt injection provenance). |
| **Planted/erroneous intermediates** | Two-hop bridge or arithmetic intermediate, patched or naturally wrong | Correct answer is text-inferable; *the model's* intermediate is not | Strongest anti-correlation with text-reasoners. G4 is built in (answer flips). |
| **Position / token count** | "Is token index > 512", "Nth item in a list" under identical local windows with varying prefix length | Local window identical; prefix padded with neutral text | Easy probe, but low salience (O3) and entangled with attention-sink/norm effects. Low priority: calibration only. |
| **Secret-knowledge models** (optional) | LoRA-trained model holding a secret it won't reveal (taboo word / hidden user attribute, Cywinski et al.-style) | Secret never appears in the prompt | Out-of-workspace by *suppression*, not access. Different claim. Fine-tuning a 27B costs ~1 day. Do it only if A works early. |

Drop: sentiment/topic/register/syntax-from-natural-text items. They're all text-inferable and are what made the original design non-discriminating.

---

## 4. Recommended design and kill criterion

**Design:** Design A (injection dose-response) as the core, Design C (planted intermediates) as the second family, and Design D's TextNLA baseline as a mandatory control column in every table.

- Out-of-workspace is defined **operationally by dose**: the α band above the probe threshold and below the self-report threshold, with self-report measured by a 5-way elicitation battery including logit 2AFC. No lens is used in the definition, so there's no circularity.
- All channels are read at **layer 42, same token**, with **K-way forced-choice scoring against matched null activations** (α=0, random equal-norm direction, other-concept injection). Report AUROC/d′, not hit rates. This handles hallucination, format asymmetry, and verbosity in one move.
- Headline figure: AUROC vs α for {probe, self-report, J-lens, Oracle, NLA, TextNLA}, with the self-report α50 marked. Second figure: Design C, P(NLA reports planted intermediate) vs P(reports text-correct intermediate).
- Interpretation table, written **before** running:
  - NLA curve left of self-report and Oracle, and TextNLA flat at chance → NLA reads sub-workspace latent content. That's a positive answer to Neel.
  - NLA curve ≈ Oracle ≈ self-report → NLA sees nothing extra here. Its benchmark advantage is elsewhere (go to D).
  - NLA right of others → NLA is worse on causal latents. Its advantage is text-driven.

**Schedule:** D1 de-risk A (5 concepts) and D (3 families). D2–3 full A sweep. D4–6 Design C. D7–9 robustness: second injection layer, in-distribution control, judge-vs-logprob scoring agreement. D10–12 write-up. Days 13–14 are buffer.

**Kill criterion (end of day 3):** pivot if **any** of these holds:
1. NLA forced-choice AUROC on *natural* (in-distribution) activations of the 30 concepts is < 0.75. The tool can't read even easy content under fair scoring, so the injection results will be noise.
2. NLA false-positive rate on null activations is so high that its AUROC is within 0.05 of chance at **every** α where self-report is at chance, *and* its α50 is not lower than Oracle's by at least a factor of ~1.5 (CI from bootstrapping over concepts).
3. No stable self-report threshold exists (self-report at chance at all α, or flat, under all elicitations). Then "workspace" isn't measurable this way.

**Pivot target:** Design D, written up as "How much of NLA's WorkspaceBench advantage is input re-description?". That means TextNLA at N tokens, claim-level precision for text-inferable vs non-inferable claims, and forced-choice rescoring of the benchmark. Keep the injection curves for J-lens/Oracle/self-report/probe as a secondary result ("the verbalization gap between probes and every verbalizer"). That's a publishable introspection-threshold result on its own, regardless of the NLA.
