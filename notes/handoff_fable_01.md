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

## Chosen design: unified source attribution (please critique hard)
For every claim each reader makes (NLA at L42; Oracle Lens at L40/44, since it was trained at layers 20–60 in steps of 4 and 42 is off its training grid; J-Lens), attribute it to one of:
1. **Prompt re-solving / re-description:** the reader rebuilds the prompt from the activation and answers or describes it.
2. **Workspace (J-space) content.**
3. **Non-workspace latent content.**
4. **Unsupported (confabulation).**

Three causal tools:
- **(A) Natural non-J features + steering.** Find features at L42 that are:
  - probe-decodable;
  - causally used, i.e. ablation changes behaviour;
  - outside J-space: the probe survives J-space removal, and J-Lens doesn't show them;
  - not self-reportable.

  Candidates: character count during line wrap, part of speech of the next token, tense. Then steer *along the probe direction* with the text unchanged. A reader that tracks the activation shifts its report; one that re-solves from text doesn't.
- **(B) Split-injection with dose-response.** Inject a concept vector split into its J-space and non-J components, at several strengths. Phase 1 controls apply: random J-space direction and isotropic random vector. Measure every channel at the same token: probe, self-report, J-Lens, Oracle Lens, NLA, and TextNLA (a text-only LLM).
- **(C) Text-identical counterfactuals.** Plant or swap an intermediate (e.g. the bridge entity of a two-hop fact) at a strength validated to *flip the model's own answer*. A faithful reader reports the planted value; a re-solver reports the original. The pilot's swap was too weak: J-Lens saw the planted bridge in only 14/50.

**Scoring:**
- K-way forced choice via teacher-forced log-probability of candidate strings under each reader, compared against matched null activations, reported as AUROC.
- Plus claim-level judged precision. The judge is validated on a human-labelled subset and must be non-Qwen.

**Headline figure:** per reader, a stacked bar of the fraction of claims attributable to prompt / workspace / non-workspace / unsupported.

**Gates already running:**
- NLA layer convention (block-41 vs block-42 output; an off-by-one risk) and NLA reconstruction-quality (FVE) reproduction.
- WorkspaceBench replication on the regex-scored families.
- Non-J feature discovery (probes, J-space removal, causal use, self-report).
- A pre-registration draft.

**Kill checks (day 2–3):**
- No natural feature passes all four non-J criteria → drop (A) and lean on (B) + (C).
- The NLA's forced-choice AUROC on natural, in-workspace content is < 0.75 → readers are too weak for attribution, so pivot to a claim-level audit.

## Questions for you: experimental-design critique first
1. **Attack the design.** Which of (A), (B), (C) is weakest or could produce a misleading answer, and why? Is the four-way attribution well-defined? Can one claim come from several sources, and how should overlap be handled?
2. **Fairness of (A).** The readers were never trained to report syntax or character counts, so they may fail for boring reasons. Is calibrating each reader on *in*-workspace features with identical scoring enough to make a non-J failure meaningful?
3. **Off-distribution risk in (B) and (C).** Injected or steered states may be out of distribution for readers trained on natural activations. How do I pick strengths? I plan to match the natural spread of the probe projection and validate by downstream behaviour change.
4. **Scoring.** Is teacher-forced log-prob forced choice fair across free-form NLA output vs four-bullet Oracle Lens output vs J-Lens token lists? Is there a better common currency?
5. **The "prompt re-solving" bucket.** I operationalise it with (a) a text-only TextNLA baseline and (b) text-identical counterfactuals. Is that sufficient, or is there a cleaner test, e.g. corrupting the activation's prompt-reconstructable component while preserving the latent?
6. **Scope.** Is three tools too much for two weeks? If you had to cut one, which?
7. **Overlap and impact.** The pilot author is probably a fellow applicant; I plan to differentiate quietly and cite them. What would make Neel pick this write-up, and what would make him stop reading?
