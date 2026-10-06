# 01: WorkspaceBench replication (regex multi-token families)

**Verdict: replicated.** All 12 published numbers (4 families × olens / nla / jlens) fall inside
our 95% bootstrap CIs. Our readers are set up correctly for the main sweep. One caveat: the NLA
numbers use the wsbench default layer convention (block-42 output). Whether NLA was trained on
block 41 or block 42 is still being checked in `experiments/00_nla_gate`; see "Dependency" below.

Date 2026-10-05. Fork `third_party/workspace-bench` (upstream `92d763e` plus the Fireworks route,
which is unused here). Judges: the regex instrument of record `mt-regex-2026-09-23`. jlens uses
`mt-regex-summarized-2026-09-23` with the Gemini 3.8 Flash summarizer via OpenRouter.

## Setup

| arm | artifact | layers produced | GPU |
|---|---|---|---|
| olens | `agu18dec/olens_and_ar:olens_s3d_rl600`, byte-identical to the gated default (subclass `OLensBatched.bind`) | 20..60 step 4 + 42 (12) | H100 |
| jlens | `neuronpedia/jacobian-lens` n1000 wikitext (downloaded to `rfm-weights` by `download_jlens`) | 20..60 step 4 + 42 | H200 (it OOMs an H100) |
| nla | `ceselder/qwen3.6-27b-nla-rl`, `av_rl_adapters/iter_000400` (wsbench default) | 42 (block-42 output) | H200 |

- **Same code as the bench.** Rendering, capture, carrier prompts, injection and sampling
  (T=1, top_p 0.95, top_k 64, 256 new tokens, k=1) all come from the fork's
  `Producer` / `OLens` / `NLA` / `JLens`.
- **One change: verbalizer generations are batched across cells.** olens uses batch 64 and nla
  batch 50. Prompts have equal length, so there is no padding, and each row samples
  independently. The bench's own batch-1 loop is about 12x slower.
- **Greedy check** (`--check`), batched vs batch-1 output: olens 2/4 identical and nla 1/2
  identical. Where outputs differ, they diverge only in the last few tokens of a 256-token
  generation. That is bf16 batch numerics, not a logic difference.
- **Judge views.** `split_readouts.py` splits the 12-layer files into views:
  - `grid` = the 11 README layers, scored best-over-layers, which is the published convention.
  - `L40`, `L42`, `L44` = single-layer files.
  - Each view is judged as-is (complete=True, no `layers=` subset flag).
  - The jlens summaries are computed once over all 12 layers and the cache is reused per view.

## Replication table (olens / jlens = best over 11 layers; nla = L42 only)

| family | arm | ours (95% CI) | README | target in CI? | n | prompt-only floor |
|---|---|---|---|---|---|---|
| typo_mt | olens | 0.86 [0.79,0.92] | 0.83 | yes | 100 | 0.67 |
| typo_mt | nla | 0.73 [0.64,0.81] | 0.75 | yes | 100 | 0.67 |
| typo_mt | jlens | 0.15 [0.08,0.23] | 0.17 | yes | 100 | 0.67 |
| multilingual_mt | olens | 0.65 [0.56,0.74] | 0.62 | yes | 100 | 0.42 |
| multilingual_mt | nla | 0.42 [0.33,0.52] | 0.44 | yes | 100 | 0.42 |
| multilingual_mt | jlens | 0.08 [0.03,0.13] | 0.06 | yes | 100 | 0.42 |
| multihop_mt | olens | 0.34 [0.24,0.43] | 0.32 | yes | 100 | 0.69 |
| multihop_mt | nla | 0.15 [0.08,0.22] | 0.22 | yes (edge) | 100 | 0.69 |
| multihop_mt | jlens | 0.10 [0.05,0.16] | 0.08 | yes | 100 | 0.69 |
| basic_readout_mt | olens | 0.73 [0.64,0.81] | 0.71 | yes | 100 | 0.59 |
| basic_readout_mt | nla | 0.57 [0.47,0.66] | 0.53 | yes | 100 | 0.59 |
| basic_readout_mt | jlens | 0.24 [0.16,0.33] | 0.24 | yes | 100 | 0.59 |

- **Floors.** The prompt-only floor comes from `evals/baselines/prompt_only.json`: stock Qwen
  sees only the text and is judged by the same instrument. The MC lucky-guessing floors are
  stale for the regex scorer and are not drawn.
- **jlens raw-bag regex** would score about 0 by construction. These jlens numbers are the
  summarized instrument.

## Matched-layer comparison (single layer, k=1 sample per cell)

| family | olens L40 | olens L42 | olens L44 | nla L42 | jlens L40 | jlens L42 | jlens L44 |
|---|---|---|---|---|---|---|---|
| typo_mt | 0.72 [0.63,0.80] | 0.76 [0.68,0.84] | 0.70 [0.61,0.79] | 0.73 [0.64,0.81] | 0.08 [0.03,0.14] | 0.07 [0.02,0.12] | 0.07 [0.03,0.12] |
| multilingual_mt | 0.26 [0.18,0.35] | 0.28 [0.20,0.37] | 0.26 [0.18,0.35] | 0.42 [0.33,0.52] | 0.00 | 0.00 | 0.00 |
| multihop_mt | 0.08 [0.03,0.14] | 0.10 [0.04,0.17] | 0.10 [0.04,0.17] | 0.15 [0.08,0.22] | 0.00 | 0.00 | 0.00 |
| basic_readout_mt | 0.42 [0.32,0.52] | 0.45 [0.35,0.54] | 0.48 [0.38,0.58] | 0.57 [0.47,0.66] | 0.04 [0.01,0.08] | 0.01 [0.00,0.03] | 0.03 [0.00,0.07] |

What this shows:
- **The headline olens > nla ordering comes from the best-over-11-layers protocol.** At a
  single matched layer, nla matches olens on typo_mt and beats it on the other three families:
  - multilingual_mt: 0.42 vs 0.28, and the CIs barely overlap.
  - basic_readout_mt: 0.57 vs 0.45.
  - multihop_mt: 0.15 vs 0.10.
  - The grid score is an any-of-11 OR over 11 independent k=1 samples, so it is partly a
    best-of-N sampling bonus as well as a layer choice.
- **L42 being off-contract does not visibly hurt olens.** olens is not trained at 42, but L42 is
  no worse than its on-contract neighbours L40 and L44: all three agree within about 0.05 on
  every family.
- **olens and jlens peak late**, at L52–60 for multilingual and basic_readout (per-layer table
  below). L42 is not either lens's best layer. A fair comparison should report both the matched
  layer and each method's best layer, with the number of samples equalised (for example
  nla k=11 at L42 vs olens any-of-11).
- **Nothing beats its prompt-only floor clearly except olens.** At L42, nla only ties the floor
  on multilingual_mt (0.42), and every arm is far below the floor on multihop_mt (0.69). Only
  olens-grid clearly beats its floor (typo, multilingual, basic_readout).

### Per-layer pass rate (item passes at that layer, from the 12-layer judge)

| family | arm | L20 | L24 | L28 | L32 | L36 | L40 | L42 | L44 | L48 | L52 | L56 | L60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| typo_mt | olens | 0.73 | 0.78 | 0.71 | 0.68 | 0.73 | 0.72 | 0.76 | 0.70 | 0.66 | 0.74 | 0.78 | 0.76 |
| typo_mt | jlens | 0.01 | 0.03 | 0.05 | 0.01 | 0.03 | 0.08 | 0.07 | 0.07 | 0.04 | 0.01 | 0.03 | 0.03 |
| multilingual_mt | olens | 0.02 | 0.03 | 0.09 | 0.10 | 0.18 | 0.26 | 0.28 | 0.26 | 0.30 | 0.39 | 0.43 | 0.38 |
| multilingual_mt | jlens | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.01 | 0.06 |
| multihop_mt | olens | 0.08 | 0.05 | 0.05 | 0.07 | 0.14 | 0.08 | 0.10 | 0.10 | 0.08 | 0.17 | 0.18 | 0.13 |
| multihop_mt | jlens | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.03 |
| basic_readout_mt | olens | 0.22 | 0.23 | 0.27 | 0.30 | 0.43 | 0.42 | 0.45 | 0.48 | 0.46 | 0.58 | 0.62 | 0.61 |
| basic_readout_mt | jlens | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.04 | 0.01 | 0.03 | 0.01 | 0.02 | 0.04 | 0.15 |

## Discrepancies and likely causes

- **No discrepancy reaches significance.** The largest gap is nla multihop_mt: 0.15 vs 0.22, at
  the upper edge of our CI. Likely causes:
  - k=1 sampling noise at n=100. The CI half-width is about 0.07.
  - Batched-generation bf16 numerics.
  - The NLA layer question below.
- **Directions are mixed.** The olens grid values are 0.02–0.03 above the README on all four
  families, and nla is up on one family and down on three. This is consistent with resampling
  noise and does not point to a systematic setup difference.

## Dependency: NLA layer convention

We ran nla at the wsbench default: the read vector is the output of block `model.layers[42]`.
`experiments/00_nla_gate` is checking whether NLA was trained on `hidden_states[42]`, which is the
output of block 41. Our nla numbers match the README, which also used the producer default. So
the README and we share whatever convention the producer uses, and the replication holds either
way. If 00 concludes block 41 is correct, re-run nla at layer 41:

    PYTHONPATH=../../infra modal run produce_modal.py --arm nla --nla-layer 41 --families typo_mt,multilingual_mt,multihop_mt,basic_readout_mt   # -> readouts/nla_L41/

That costs about 5 GPU-min. The re-run might raise nla's numbers.

## Throughput and memory (for planning the main sweep)

| method | GPU | batch | s / cell (steady) | load | peak GPU mem |
|---|---|---|---|---|---|
| olens (gen ≤256 tok) | H100 80GB | 64 | 0.49–0.57 | 31–47 s | 66.0 GB |
| olens, batch-1 (bench loop, smoke) | H100 | 1 (24 cells) | ~1.2 at batch 24; batch-1 ~10+ (est.) | — | 58.7 GB |
| nla (gen ≤256 tok) | H200 | 50 | 0.68–0.72 (first family 1.35, incl. warm-up) | 65–87 s | 119.2 GB |
| nla, batch-1 (smoke) | H200 | 1 | 12.6 | — | 111.5 GB |
| jlens (top-10, unbatched) | H200 | 1 | 0.165 (0.21 on the first family) | 86 s | 126.7 GB |

- Capture is negligible: 0.4 s for 2 items, about 4 ms per item.
- **olens.** One item at 12 layers takes about 6 s.
- **nla.** There is room for a batch of about 64 on an H200.
- **jlens.** It is not batched. Batching the matmul, or keeping J only for the needed layers in
  bf16, would make it about 10x faster and let it fit on an H100. Its peak memory is mostly J
  plus W_U in fp32, plus the `W_U @ J_L` transient.

## Costs

- **OpenRouter: $2.80** in total, all jlens summaries (4,836 calls = 4 families × 1,200 cells +
  36 smoke). Gemini 3.8 Flash costs about $0.00058 per summary call. The key's `usage` went from
  $0.00006 to $2.80, which matches the reported `spend_usd` (no ~2.5x under-reporting seen on
  this call type). This is well under the $15 cap. Prose arms cost $0 (regex).
- **Modal GPU (estimated from wall-clock):**
  - olens: about 4 × 12 min H100.
  - jlens: about 18 min H200, plus a failed 3-min H100 attempt.
  - nla: about 8 min H200.
  - smokes: about 12 min.
  - Total about 1.5 GPU-h, roughly $6–7 at list price.

## Files

- `produce_modal.py`: Modal app `mats-01-wsbench-replication`. Contains `download_jlens`,
  `produce_h100` / `produce_h200`, the batched `OLensBatched` / `NLABatched` classes, and the
  `--check` greedy comparison.
  - Usage: `PYTHONPATH=../../infra modal run produce_modal.py --arm olens --families typo_mt --per-family`
  - It builds its own image with `modal_common.gpu_image`'s pins. That image cannot be
    extended, because it ends in a local mount.
- `split_readouts.py`, `judge.sh`, `summarize.py`: split views, judge with the fork, write the
  tables.
- `readouts/<arm>/<family>.jsonl` (4.4 MB): 12-layer olens/jlens and L42 nla. The same files are
  on the volume at `/results/01_wsbench_replication/readouts/`.
- `judged/<arm>_<view>/<family>/results.json`, plus the `cells.jsonl` summarizer cache, so
  re-judging jlens is free.
- `meta/*.json`: per-run throughput and peak memory.
