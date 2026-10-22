# Paste-ready text for the shared Doc ("When activation readers agree, are they right?")

Insertion points are marked `>>>`. Everything else is text to paste as-is. Written in the Doc's voice (merged "we"). Numbers trace to `experiments/*/RESULTS.md` and `notes/decisions.md`; pilot numbers are marked as such.

---

>>> TITLE. Suggest: **When activation readers agree, are they right? And do they read what the model can't say?** (The Doc is currently "Untitled document".)

>>> TLDR. Add after the fourth bullet:

•       Nothing we can certify as outside the J-lens's "workspace" is described by any reader. The usual test for non-workspace content, "does a probe survive removing the J-space?", passes every feature we tried, topic included. A calibrated gate leaves one clean feature on Qwen at layer 42, the number of characters since the last newline. The model uses it (steering it moves the line-break decision 49× more than random directions), and when we move it inside the activation with the text held fixed, no reader's description follows, at a dose where the same readers track language and tense at z +20 to +30. Two narrow channels are candidates for a pre-registered run that starts 19 October.

•       What the NLA adds is not non-workspace content. It generates: with a different fact planted in the activation, it rewrites the prompt to fit the plant, 12 times out of 12. On WorkspaceBench, a text-only model given the prompt scores 0.89 against 0.60 and 0.56 from the NLA and Oracle readouts, and the Oracle Lens's published lead is a best-of-11-layers scoring artefact.

>>> SETUP. Add as a second paragraph:

For the non-workspace half we read the same site with the same readers, plus a second J-lens (the 25-prompt camilablank lens, which differs a lot from the neuronpedia one: vector cosine 0.79, top-25 overlap 19%) and, for one experiment, the J++ lens. By "J-space" we mean the top-r right singular vectors of W_U·diag(1+w)·J₄₂. We use r\* = 384, the smallest rank at which a rebuilt J-lens keeps 90% top-10 agreement with the full one, and 4r\* = 1,536 to be safe. Features are moved inside the activation with the text fixed, h′ = h − (v·h)v ± Δv, equal norm on both sides (the additive version produced a spurious Oracle "anti-tracking" whenever h·v ≠ 0, so we dropped it). Doses come from the natural spread of the feature, and every reader statistic is a z against random directions of the same norm, drawn both isotropically and from the complement of the J-space. A dose is thrown out if the J-lens itself moves under it.

>>> NEW SECTIONS. Insert after section 4, before Misc.

## **5. The usual test for "non-workspace" content passes everything**

The published way to show that a feature is outside the workspace is to train a probe, project out the J-space, retrain, and check that accuracy survives. We ran that on eight features at layer 42: characters since the last newline ("col"), the same count asked for explicitly, bracket depth, the model's predicted part of speech, tense, language, topic and next-token identity. All eight keep at least 98% of their probe score after removing the top-384 or top-1,536 directions, topic and language included (topic 0.849 → 0.842 / 0.834). Removing ten random subspaces of the same rank gives the same retention within ±0.02. Even at rank 4,000 of 5,120 the extra drop beyond random is at most 8% of the above-chance margin. A feature that is strongly represented is decodable from almost any subspace of a 5,120-dimensional residual, so the test measures probe redundancy, not where the feature lives.

We replaced it with four gates, each against a matched random null. G1: a probe beats chance and a control task where each token type gets a random label. G2: the feature's class-mean direction has no more energy in the top-r J-space than an ordinary activation direction (enrichment ≤ 1.5 at both ranks; random = 1.0; topic scores 5.98, tense 5.60, language 3.32). G3: the tokens the J-lens actually emits don't track the feature's value, scored in the form the lens would use (digits for counts, the part of speech of the top-10 for POS) against a permutation null. G4: steering along the direction with its J-component removed changes the model's own behaviour more than random directions do, without the J-lens moving.

One natural feature passes all four: col. It is J-depleted under both lenses (enrichment 0.67 / 1.10 and 0.82 / 1.41), the lens never emits its value (AUROC 0.500, band [0.499, 0.501]), and the J-removed direction shifts the model's newline log-odds by +0.99 nats against 0.02 for random directions. It also keeps 79% of its causal effect after J-removal, where tense keeps 19% and language 5%. Bracket depth is readable through bracket tokens. Predicted POS looked unreadable until we classified the lens's own next-word guesses by part of speech, at which point it read at 0.66. Nine more candidates (capitalisation, quote state, code language, French gender agreement, indentation, words since sentence start, list index) all failed somewhere; indentation is the interesting failure, decodable and J-depleted but causally inert. Our one-line summary, which is a survey and not a theorem: on this model at this layer, the non-workspace content the model actually uses is positional bookkeeping. Numbers never enter the J-space here even when the model is asked to report them, so we have no in-workspace count to use as a control. That is a limitation we state rather than fix.

## **6. No reader describes it, at a dose where they read workspace features easily** (pilot)

The calibration first. With raw language and tense directions at the same relative norm, the NLA tracks language at AUROC 0.87 (z +30) and tense at 0.92; the Oracle tracks language at 0.80 (z +20) and, with templates mined from its own output, tense at 0.99. The J-lens reads both. So the method sees workspace features at the dose we then use for col.

On col, with descriptive templates ("near the end of a long line" and the like), every reader stays inside the random band: NLA 0.50 to 0.53, Oracle 0.49 to 0.51, every z within ±1.5, across every template family we tried. None of 5,760 free readouts mentions line length or position. Meanwhile the same edit moves the model: newline log-odds +0.72 (z +55), and greedy continuations wrap 1.55 characters earlier. The J-lens stays flat, which is what makes the edit a fair test, though only once the top 1,536 directions are removed; removing 384 leaked end-of-line tokens into the lens.

Two things did move, with phrasings taken from the readers' own output rather than ours. The Oracle's next-token continuation, "- \nthe" against "- the", reaches 0.67 to 0.72 (z ≈ +3.9) and collapses with the LoRA off. One NLA phrasing, "mid-sentence continuation located near the end / beginning of a hard-wrapped line", reaches 0.63 (z +3.5), but it is the best of eleven templates and we expect it to shrink. Both are pre-registered as primaries for the main run, together with the two tense cells from section 8, with Holm correction, fresh items, and a null band of 100 directions. We had earlier written "no reader reads col" and retracted it when these two turned up; the honest statement is "no robust descriptive reading, two narrow candidate channels".

## **7. The Oracle's channel moves with the model's own next-token decision, mostly** (pilot)

If the Oracle Lens is the model plus a LoRA, its continuation channel might just be re-enacting what the model would do next. We measured the gain ratio, the Oracle's continuation shift divided by the model's own behavioural shift at the same edit. On col it is 0.99 [0.83, 1.20] for the raw direction and 1.06 [0.89, 1.26] after J-removal, with the same retention under J-removal as the model itself (0.79 vs 0.75). Random directions that happen to have a behavioural effect sit at 1.65 to 1.89. That looks like acting rather than reading.

But it isn't the whole story. The Oracle also tracks the J-removed tense direction (z +12 to +16) where the model's own next-token tense readout does not move (−0.04 nats), and shows a dose-dependent response to indentation, which the model ignores. So the channel carries more than the model's next decision, and we demoted our planned binary verdict to three descriptive numbers. This matches section 2 from the other direction: the Oracle reports what the vector contains, whether or not the model is about to use it.

## **8. What the readers do pick up: the tail of what the model can say, and what it doesn't use**

Tense has a component outside the top-1,536 J-space, and both decoders read it: Oracle 0.66 to 0.79 (z +12 to +17), NLA 0.59 to 0.74, leak-free. With our first fixed templates only the NLA moved and we briefly read that as an NLA-over-Oracle contrast; mined templates removed it. Tense is self-reported 87% of the time, so this is the distributed tail of a feature the model can say, not something it can't.

Planting a fact gives the same picture as section 2. We swapped the bridge entity's class-mean projection at layer 38 ("the country famous for the Eiffel Tower" → Sweden) with the text unchanged, keeping only items where the model's answer flips, unrelated prompts stay put, and the prompt is still decodable at layer 42 (431 of 614). All three readers report the planted country: Oracle 1.00, NLA 0.98, J-lens 1.00, from 1 to 3% unplanted. The swap direction is 6.5× J-enriched, so this is in-workspace calibration. The part worth keeping is what the NLA writes: "…the country whose CEO designed adopted Volvo… Sweden", "…maple leaves (Canada)". In 12 of 12 free explanations it names the planted country and rewrites the clue to match. Its "prompt reconstruction" is generated from the latent, not read from the text.

Michael Kenney's arithmetic data gave us one more case. On 600 "a ± b =" prompts a linear probe reads both operands at the "=" position (83% and 93% exact), while the NLA reports them 5% and 38% of the time. We replicated the probe numbers and ran the gate: the operand digits are J-depleted under both the neuronpedia and J++ bases, and the J-lens never emits a digit. But on 117 one-digit few-shot items, no leak-clean J-removed edit at the pre-registered doses changed a single greedy answer (log-odds shifts under 0.1 nats against a 6.65-nat margin), and replacing the whole vector at "=" with another item's flips 1 of 117. The answer-determining operand information is not at that position at layer 42; the model reads the operands from their own tokens later. Decodable, outside the workspace, and unused, exactly the cell section 2 found for ownership. Kenney's own steering split (readers follow class-mean directions and ignore probe-weight directions) is the same finding in a third dataset.

## **9. Benchmarks: the prompt explains the readouts**

All 12 published WorkspaceBench numbers (four regex families × three readers) reproduce inside our 95% intervals. The Oracle's lead comes from counting an item as solved if any of 11 layers gets it, worth +0.08 to +0.22; at layer 42 alone the NLA ties or beats it. On 400 items, every item either prose reader gets right is also reachable from the prompt text: a text-only base Qwen writing in the NLA's register scores 0.46 against the NLA's 0.47 (p = 0.75); given only the NLA's own readout a text-only re-solver scores more than the NLA does (0.57 to 0.60); given the prompt, 0.89; given prompt plus readout, 0.88. Claims the readers make are text-true less often than a text-only model's (precision 0.62 and 0.52 against 0.92). As scored, the benchmark cannot separate reading the activation from re-solving the prompt. A side finding we have not explained: its frozen prompt-only floors look about 0.14 too low on three of four families.

## **Putting the two halves together**

**Three findings that stand regardless of how the main run comes out:**

1. "Not in the workspace" has no off-the-shelf test. The probe-survival test passes every feature we tried, in-workspace references included, by the same margin as removing random directions (section 5).
2. On Qwen3.6-27B at layer 42, the natural content that is outside the J-space and causally used is positional bookkeeping. col passes all four gates under two lenses, nine further candidates fail, and numbers never enter the J-space even on request (section 5).
3. WorkspaceBench cannot separate reading an activation from re-solving the prompt. The prompt alone scores 0.89 against 0.60 and 0.56 from the readouts, and 0.88 with both (section 9).

Three things show up on both sides. Readers report what is linearly present along class-mean-like directions, whether or not the model uses it (planted ownership at the last token; indentation; operand digits at "="). Where the prompt already states the fact, agreement among readers tells you nothing a prompt-only baseline wouldn't (32/32 in section 1; 0.89 in section 9). And the readers behave as one instrument: they track the same workspace features at the same dose, fail together on the one certified non-workspace feature, and share wrong answers twice as often as chance.

So, to the question we started from. Can agreement among readers stand in for ground truth? No: it is circular when the readers share a base model, uninformative when the prompt states the fact, and inverted on the model's own errors. Do NLAs read what the model can't say? Not that we can find at this site, pending the main run; what the NLA adds is a story generated from the latent, not access to non-workspace content. The ground truth we would trust is built without the readers: is the fact linearly present (probe with a control task), is it load-bearing (an edit flips behaviour against random-direction nulls, with the J-lens checked for leaks), and is it inside or outside the J-space (the gate in section 5). Readers are then rated on coverage of present-and-used content, on false positives for present-but-unused content, and on how well their shift predicts the model's shift under intervention. Our steering data already has that form.

>>> MISC. Remove the two "Suraj's companion project" bullets (now sections 8 and 9). Keep the rest.

>>> WHAT WE'D DO NEXT. Add after "SAEs are still not in the comparison.":

For the non-workspace half: the pre-registered main run (19 October to 1 November) decides whether the two narrow channels survive on fresh items. SAE features active at col positions can go straight through the gate (decoder-direction enrichment, then the same steering test), which is the cheapest way to test whether SAEs find non-workspace content. And since the "=" position does not carry the operand information the model uses, a causal test of operand reading needs edits at the operand tokens or across layers, which is the same site-finding problem as the error set.

>>> CAVEATS. Add one sentence to the existing caveats paragraph:

The non-workspace steering numbers are from a pilot split of 60 items per feature with 20-direction null bands, excluded from the main analysis; one clean non-workspace feature, and it is a count.

>>> QUESTIONS. Add:

4.   Is "outside the J-space, decodable, and unused" a category worth naming, given it now shows up in three datasets (ownership at the last token, indentation, operand digits)? If readers faithfully report it, a reader can be right about the vector and useless about the model at the same time.

5.   Should a reader be scored on what it decodes from clean prompts, or on how well its shift predicts the model's shift under intervention? The two rank readers differently in our data.

>>> REFS. Add:

Blank et al. 2026 – WorkspaceBench – github.com/camilablank/workspace-bench
Ayonrinde & Lindsey 2026 – J++ Lens
Kenney 2026 – nla-consistency – github.com/mkenney2/nla-consistency
Hewitt & Liang 2019 – Designing and Interpreting Probes with Control Tasks – https://arxiv.org/abs/1909.03368

---

>>> NEW SECTION for the shared Doc, after section 9 (added 13 Oct after Agam's comments). Pilot / exploratory throughout.

## **9. Robustness checks (pilot)**  (section 9 in the current Doc, after "8. Benchmarks", before "Misc")

Six checks requested by our mentor after reading sections 5–9, run on the pilot split; nothing here changes a pre-registered cell.

**Oracle Lens off its trained layers.** The Oracle was trained on every fourth layer and 42 is between two of them. Rescoring its continuation channel on the J-removed col push with the pilot's own directions and null vectors: at the trained layer 40, AUROC 0.65, paired win 0.98, +1.48 nats, z +4.8; at 42, 0.67 / 0.98 / +1.69 / z +5.3. Layer 44 is void under the leak rule (its smaller J-space removal leaves the newline readout in the J-Lens, z +2.6). On the descriptive templates the three layers are indistinguishable (AUROC 0.46–0.53). Across the Oracle's grid 36–52 it reads the J-removed tense tail at every layer (z +5 to +10, leak-clean). The NLA cannot be moved: its injection is fixed to block 42.

**Where "non-J" holds.** A layer-by-position map (layers 24–56, the read token and three before it, class-mean directions per cell) shows the col direction 3.7× J-enriched at layer 24 and 2.2× at 36, falling to about our 1.5 cutoff only at 42–48; the J-removed push is leak-free to 48 and leaks into the J-Lens at 52–56, so col qualifies as non-J only in a band of about 42–48. "Outside the workspace" for col is a property of the band around the readers' layer, not of the model. `[comment: per-cell probes here are 60/120-item cross-validated fits and are noisy; exp 02's 40k-row probes give the decodability numbers. The sweep's class-mean directions have cosine ≈ 0.4 with the pilot's bank directions, so the sweep understates every reader; the 40/42 numbers above use the bank directions.]`

**Sampling and consensus.** On the stored k = 4 readouts no sample mentions line position in any condition (0–0.8%), so consensus has nothing to amplify. With k = 8 at the P1/P2 cell and a judge counting support per claim: 2 of 4,320 samples make a line-position claim, no item has two supporting samples, and the consensus rate is 0/60 for every reader and condition. Eight draws and a judge confirm the single-sample null; the pilot's candidate signal is in the template log-odds, not in anything the readers say. The main run's template-free check moves to k = 8 and reports the consensus rate beside the per-sample rate (rev 4.3).

**Punctuation.** Stratifying by the context at the read token and by the model's next-token punctuation mass: the Oracle's continuation channel on col is flat across strata (1.72 nats everywhere), its sentence-template tense channel is flat (1.34 vs 1.14 nats), and the model's own next-token tense readout is flat in every stratum. Only the bare tense-token channel ("- was" vs "- is") is boundary-driven, about 2× at sentence starts.

**Same hallucinations?** On 100 WorkspaceBench items the NLA has a text-contradicted claim on 97 and the Oracle on 77; on the 75 where both do, a judge finds the same wrong content on 16% (0% for mismatched items). On arithmetic, both prose readers write an expression and both misquote the operands (Oracle correct on 7 / 120; NLA 5% / 38% per operand), while the J-Lens never emits a digit. The readers share the kind of error far more than the content.

**Side by side.** Readouts of the same activation by all three readers (WorkspaceBench items, col at zero / ±Δ, the 12 planted-bridge pairs with Oracle bullets at 42 and 44, arithmetic prompts) are in an appendix; the Oracle names the planted country in 96% of planted items and 0% of natural ones at both layers.
