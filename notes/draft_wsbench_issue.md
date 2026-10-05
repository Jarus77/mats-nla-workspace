# DRAFT: GitHub issue for camilablank/workspace-bench (NOT POSTED; needs the user's approval)

**Title:** Frozen mt prompt-only floors appear ~0.14 low under the thinking-off rendering (re-run table attached)

---

Hi, thanks for releasing WorkspaceBench. We've been using it for a MATS project and are grateful it's so well documented.

While building text-only baselines, we re-ran the prompt-only floor for the four regex-scored multi-token families. We got noticeably higher numbers than the frozen values in `evals/baselines/prompt_only.json`. We don't know the cause and may well be missing something about how the floor was generated, so we're sharing the re-run in case it's useful.

**Setup.** Base `Qwen/Qwen3.6-27B` (snapshot `6a9e13bd`), the template and sampling recorded in `_source` (k=1, T=1.0, top_p 0.95, top_k 64, max_new_tokens 256, no stop, empty output_prefix), seed 0, scored with the repo's regex scorer (`mt-regex-2026-09-23`), n=100 per family.

| rendering | typo_mt | multilingual_mt | multihop_mt | basic_readout_mt |
|---|---|---|---|---|
| frozen floor | 0.67 | 0.42 | 0.69 | 0.59 |
| chat template, `enable_thinking=False` | 0.83 | 0.47 | 0.84 | 0.80 |
| (independent re-draws, same rendering) | 0.83–0.85 | 0.51–0.58 | 0.79–0.88 | 0.77–0.80 |
| chat template, thinking on (thinking text kept) | 0.64 | 0.59 | 0.78 | 0.54 |
| raw text, no chat template | 0.80 | 0.45 | 0.79 | 0.65 |

**Observations:**
- **The thinking-off re-run is higher than frozen.** Under the thinking-off chat rendering (the one `docs/read_sites.md` describes for the banks), our re-run is +0.05 to +0.21 above the frozen floors. That's significant on typo, multihop and basic_readout in two independent draws.
- **No rendering we tried reproduces the frozen values exactly.** Thinking-on and raw rendering come closer (mean |Δ| ≈ 0.08).
- **Empty cells:** the frozen run has a few empty cells (typo 3, multilingual 4, multihop 1, basic 1). None of our 1,200 re-run generations is empty. That made us wonder whether a serving-side difference is involved, but we couldn't check, since the generator and summaries aren't public.

If it's helpful, we're happy to share our generation script and outputs. And if we've misread how the floor is meant to be produced, we'd appreciate a pointer.

Thanks again!
