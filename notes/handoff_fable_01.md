# Handoff #1 to Fable (research guide) — 2026-10-05

Paste everything below the line into the Fable 5.1 chat.

---

Hi Fable. I'm running the engineering and research side of my MATS 12.0 (Neel Nanda stream) sprint in Claude Code, and I want your read as my research guide. Where we are first, then specific questions. Please be blunt; I'd rather kill an idea on day 3 than day 12.

## Status
- **Problem:** Neel's "NLAs seem to capture more than J-lens/Oracle Lens. Is that really the non-workspace stuff?" on Qwen3.6-27B. All three readers are public and verified:
  - NLA: `ceselder/qwen3.6-27b-nla-rl`, layer 42 only.
  - Oracle Lens: the WorkspaceBench s3d-rl600 adapter.
  - J-Lens: `camilablank/workspace-lenses`.
- **Infra working:** weights on a Modal volume (175GB); an H100 smoke test loaded Qwen3.6-27B, read the L42 residual and generated text, peaking at 54GB of GPU memory. Judge keys: Gemini 3.8 Flash via OpenRouter for comparability with published numbers, and Fireworks (non-Qwen judges) for our own work.
- **I'm full-time and chose "go bold".** My Phase 1 code and data were recovered.
  - Phase 1 = injecting J-lens concept directions into Qwen3-8B and eliciting self-report.
  - Yes/no detection was mostly non-specific: random J-space direction 0.79 vs concept 0.83.
  - Naming was concept-specific: 21.6% top-5 vs 0 for an isotropic random vector.
  - Colours were present in the lens but never named (0/200).

## Novelty check: what's already done (this changed the plan)
1. **A concurrent MATS-12 pilot on exactly this question:** github.com/LittleMeHere/nla-vs-lenses, created Oct 4. Same model, same readers, WorkspaceBench items.
   - Multihop at L42: J-Lens names the hidden bridge in 39/100, Oracle Lens 80, NLA 75.
   - Deleting the bridge's J-Lens directions: J-Lens 15 → 0, Oracle Lens 30 → 17, NLA 36 → 25. They conclude the readers mostly *re-solve* the question from the prompt they decode.
   - Their swap test (plant another item's bridge) was weak: J-Lens shows the planted bridge in only 14/50.
   - Chain arithmetic at L56: Oracle Lens 36/60 (chance 12); NLA near chance.
   - Their summary: "On no task does the NLA find a hidden step the oracle lens misses."
   - They explicitly list as **not tested:** "content known to be outside J-space". Their stated next step is split-injection (a concept split into J and non-J parts).
2. **Viswanath, LessWrong, Jul 2026 (Llama-3.3-70B):** concepts injected after splitting into J and non-J parts. The model names the J part 80% and the non-J part 0%; the NLA names both 100%. No Oracle Lens. On natural line counting, the NLA did *not* read the character count.
3. **Hong, LessWrong, Jul 2026 (Qwen3.6-27B):** about 95% of the NLA's reconstruction gain during RL is outside J-space, but about 90% of its specific claims are unsupported.
4. **The J-space paper** reports features used but never in J-space: part of speech and tense (probes survive J-space removal), and character count during line wrapping. That gives natural ground truth for non-workspace content.
5. **On the NLA's training data:** its warmstart explanations were written by Sonnet from the text prefix alone, with no activations. About 60% of its multihop concepts are prompt wording or format.

My two earlier candidate designs, injection dose-response and planted intermediates, overlap the pilot and Viswanath. So the plan moved.

## Current proposal: natural non-workspace features, made causal
- **Ground truth without the readers being compared.** Find features at L42 on *natural* text that are:
  - (i) linearly decodable by a probe;
  - (ii) causally used: ablating the probe direction changes behaviour (e.g. line-break placement, agreement, tense);
  - (iii) outside J-space: the probe survives removal of J-space, and J-Lens doesn't show them;
  - (iv) not self-reportable when the model is asked, across several elicitation formats.
  - Candidates: character count since the last newline (line wrapping), part of speech of the next token, tense, syntactic depth, and possibly token position.
- **Test 1 (natural).** Do J-Lens, Oracle Lens and the NLA report the feature value?
  - Scoring: K-way forced choice against matched null activations, reported as AUROC.
  - Baseline: TextNLA, a text-only LLM given the prompt.
- **Test 2 (decisive, causal).** Steer *along the probe direction* (e.g. shift "character count" by +k) with the text unchanged.
  - A reader that tracks the activation shifts its report.
  - A reader that re-solves from the decoded text doesn't.
  - This separates "reads the latent" from "re-describes the input", the confound that undermined the pilot's multihop results.
- **Predictions.**
  - If Neel's suspicion is right, the NLA reports non-J features and tracks the steering while Oracle Lens and J-Lens don't.
  - If wrong, no reader tracks them, so the NLA's extra content is prompt restatement plus confabulation.
  - Either outcome is a clean answer.
- **Kill check (day 2–3):**
  - At least one feature passes (i)–(iv) on Qwen3.6-27B.
  - The NLA gives a non-trivial forced-choice AUROC (> 0.6) on at least one feature naturally.
  - If no feature passes, pivot to a claim-level audit of the NLA's WorkspaceBench advantage (prompt restatement vs confabulation vs verified).

## Questions for you
1. Does the natural-non-J-feature design answer Neel's question more directly than split-injection? Or is it too indirect, since the readers were never trained to report syntax or character counts and so would fail for uninteresting reasons?
2. **Coordination.** The pilot author is probably a fellow applicant on the same problem. Should I reach out to divide the territory (they take split-injection, I take natural non-J features), stay quiet and differentiate, or switch problems? How does Neel tend to view overlap?
3. **Fairness.** How do I make "the NLA fails to report part of speech" mean something? Should I first calibrate each reader on *in*-workspace features with the same scoring, so a failure on non-J features is relative?
4. **Scoring.** For prose readers, I plan teacher-forced log-probability of K candidate value strings, converted to AUROC. Is that fair across a free-form NLA and four-bullet Oracle Lens output?
5. **Steering scale.** In Test 2, how far should I steer along the probe direction before it's off-distribution for the readers? I plan to match the shift to the natural spread of the probe's projection.
6. What would make Neel pick this write-up, and what would make him stop reading?
