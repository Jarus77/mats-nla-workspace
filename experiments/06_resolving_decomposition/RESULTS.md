# 06: re-solving decomposition on WorkspaceBench (natural arm, H4, panel 2)

**Verdict.** At L42, on the 4 regex multi-token families (n=400 items), *every* item either prose reader gets right is also reachable from the prompt text alone.
- NLA@42 has 187 hits; 0 of them are missed by every text-only arm.
- OL@42 has 159 hits; 4 are missed by every text-only arm. All 4 are multilingual items scored through Chinese forms of the target. That looks like a scoring-form artefact, not latent content.
- A text-only base Qwen3.6-27B, writing in the NLA's own register, scores **0.46** pooled. NLA@42 scores 0.47 (McNemar p=0.75) and OL@42 scores 0.40.
- Given only the NLA's own readout, a text-only re-solver scores *more* than the NLA itself: 0.57 (Qwen) / 0.60 (Gemini). The same holds for OL: 0.53 / 0.56 vs 0.40.
- **Text-verifiable precision of readout claims:** NLA 0.62, OL 0.52, text-only TextNLA 0.92.

**What this means for H4.** On this benchmark, success is fully text-explicable at the item level. WorkspaceBench mt regex scores therefore cannot separate reading the activation from re-solving the prompt. This is not evidence that the readers *don't* read the activation: the L42 activation encodes the prompt, and reading it then re-solving looks identical here. It supports the panel-2 / P2 framing and the need for the text-fixed steering arm.

Date 2026-10-05. Model Qwen3.6-27B. Readouts: the reader cells from `experiments/01_wsbench_replication/readouts`, k=1 sample per cell. Readers:
- NLA iter_000400 @42;
- OL `olens_s3d_rl600` @40/42/44 plus its grid;
- J-Lens @42.

Scorer: the fork's regex instrument `mt-regex-2026-09-23`, run with `uv run wsbench run` on every arm, unchanged.

## Arms

| arm | input | model | template | k |
|---|---|---|---|---|
| **T_qwen** (primary) | item prompt | base Qwen3.6-27B (Modal H200, no activation) | NLA warm-start instruction (`av_base/nla_meta.yaml`, `data/nla_instruction.txt`) | 3 (s0 = k-matched primary) |
| T_wsb | item prompt | base Qwen | wsbench prompt-only floor template | 3 |
| T_gem | item prompt | Gemini 3.8 Flash (OpenRouter, reasoning minimal) | NLA instruction | 1 |
| T_gem_wsb | item prompt | Gemini 3.8 Flash | wsbench prompt-only template | 1 |
| T_qwen_k3 | – | – | any of T_qwen s0–s2 | (3) |
| T_any | – | – | union of all 8 prompt-only samples (2 models × 2 templates) | (8) |
| Resolve(NLA) | **only** the NLA@42 readout | Qwen (k=3, s0 used) / Gemini | `RESOLVE_TEMPLATE` in `common.py` (family-neutral: reconstruct, name the entity/term spelled correctly, state the language) | 1 |
| Resolve(OL@42) | **only** the OL@42 readout | Qwen / Gemini | same | 1 |

All arms use the readers' sampling settings: T=1, top_p 0.95, top_k 64, 256 new tokens (Gemini: max_tokens 256), thinking off. Items are plain-render and read at the final token, so the "prefix" is exactly `items.json` `prompt`. I verified this against the fork's read plan for 400/400 items.

## 1. Pass rates (L42, k=1 unless noted; 95% item-bootstrap CI)

| arm | typo_mt | multilingual_mt | multihop_mt | basic_readout_mt | pooled (n=400) |
|---|---|---|---|---|---|
| NLA@42 | 0.73 [0.64,0.82] | 0.42 [0.32,0.52] | 0.15 [0.09,0.22] | 0.57 [0.47,0.67] | 0.47 [0.42,0.52] |
| OL@42 | 0.76 [0.67,0.84] | 0.28 [0.19,0.37] | 0.10 [0.05,0.17] | 0.45 [0.35,0.55] | 0.40 [0.35,0.45] |
| OL@40 / OL@44 | 0.72 / 0.70 | 0.26 / 0.26 | 0.08 / 0.10 | 0.42 / 0.48 | 0.37 / 0.39 |
| JLens@42 | 0.07 | 0.00 | 0.00 | 0.01 | 0.02 |
| OL grid (any-of-11, published protocol) | 0.86 | 0.65 | 0.34 | 0.73 | 0.65 |
| **T_qwen** | 0.45 [0.35,0.55] | 0.40 [0.30,0.50] | 0.44 [0.34,0.54] | 0.53 [0.43,0.63] | **0.46 [0.41,0.51]** |
| T_qwen per sample s0/s1/s2 | 0.45/0.49/0.50 | 0.40/0.42/0.36 | 0.44/0.43/0.46 | 0.53/0.59/0.58 | |
| T_qwen_k3 | 0.68 | 0.65 | 0.64 | 0.81 | 0.69 |
| T_wsb | 0.85 [0.78,0.92] | 0.58 [0.48,0.68] | 0.82 [0.74,0.89] | 0.77 [0.69,0.85] | 0.76 [0.71,0.80] |
| T_gem | 0.66 [0.56,0.75] | 0.73 [0.64,0.82] | 0.75 [0.66,0.83] | 0.87 [0.80,0.93] | 0.75 [0.71,0.79] |
| T_gem_wsb | 0.90 | 0.28 | 0.77 | 0.55 | 0.62 |
| T_any (8 samples) | 1.00 | 0.87 | 0.98 | 0.98 | 0.96 |
| Resolve(NLA) Qwen / Gemini | 0.75 / 0.77 | 0.61 / 0.63 | 0.26 / 0.32 | 0.65 / 0.68 | **0.57 / 0.60** |
| Resolve(OL@42) Qwen / Gemini | 0.72 / 0.73 | 0.61 / 0.59 | 0.18 / 0.28 | 0.60 / 0.64 | **0.53 / 0.56** |

**The template matters a lot.** For the same model, the NLA register gives typo 0.45 and the wsbench template gives 0.85. Gemini on multilingual goes 0.73 → 0.28. "Text explains it" is therefore a claim about what the text makes *available*. It is robust only under a union of templates, which is why T_any exists.

**The frozen prompt-only floor is too low.** Our T_wsb uses the frozen floor's exact template and sampling. It scores 0.85/0.58/0.82/0.77 against `evals/baselines/prompt_only.json` 0.67/0.42/0.69/0.59 (+0.13 to +0.18). The cause is inconclusive; see "Prompt-only floor verification" at the end. Either way, published readers vs "prompt-only floor" understates how text-available these targets are.

## 2. Item-level decomposition (pooled n=400; per-family tables in `data/tables.md`)

"Reader-only" = the reader passes and the text arm fails. "Expected under independence" = within-family product of marginals; it shows whether reader hits sit on text-easy items.

| reader | text arm | reader hits | P(text ✓ ∣ reader ✓) | reader-only (obs / exp. indep.) | text-only | McNemar p |
|---|---|---|---|---|---|---|
| NLA@42 | T_qwen (k=1, NLA register) | 187 | 0.57 | 80 / 100.5 | 75 | 0.75 |
| NLA@42 | T_qwen_k3 | 187 | 0.82 | 34 / 54.3 | 125 | 2e-13 |
| NLA@42 | T_gem (k=1) | 187 | 0.82 | 33 / 47.3 | 147 | 2e-18 |
| NLA@42 | T_any | 187 | **1.00** | **0** / 6.9 | 196 | 2e-59 |
| OL@42 | T_qwen | 159 | 0.55 | 72 / 85.3 | 95 | 0.088 |
| OL@42 | T_qwen_k3 | 159 | 0.80 | 32 / 46.3 | 151 | 1e-19 |
| OL@42 | T_gem | 159 | 0.79 | 34 / 41.8 | 176 | 3e-24 |
| OL@42 | T_any | 159 | **0.97** | **4** / 4.7 | 228 | 4e-62 |

Per family, k-matched (T_qwen):
- **typo is the only family where readers beat text.** NLA 73 vs 45 (35 reader-only vs 7 text-only, p=1.5e-5); OL 76 vs 45 (p=1.6e-6). The gap closes with T_qwen_k3 (p=0.5 / 0.23) and reverses with the wsbench template (0.85) or Gemini wsb (0.90). The NLA-register TextNLA writes ~80–100 words and quotes the misspelling without spelling the correction (54 of its 55 typo failures quote the misspelled word). So the typo gap is a register/length effect, not reader-only content.
- **multihop: text crushes readers.** NLA 15 vs T_qwen 44 (p=1e-6), vs T_gem 75. All 15 NLA multihop hits are also T_gem hits.
- **multilingual / basic_readout:** NLA ≈ T_qwen (p=0.87 / 0.63); NLA < T_gem (p<1e-6).

**Hard reader-only hits** (no text-only sample out of 8 hits the target):
- NLA@42: **0/187**.
- OL@42: **4/159**, all multilingual (sandpaper, hailstone, syringe, clothes iron).
- OL grid: 6/258, all multilingual.

The OL hits are in OL's habitual Chinese code-switching (砂纸, 冰雹, 注射器, 熨斗), which match the items' `zh` forms. The text arms named the concept in other forms that the multi-token filter does not credit. Reader hits sit on text-easy items: observed overlap is above independence in every row.

## 3. Re-solving ceiling (text-only LLM sees only the reader's readout)

| reader | re-solver | reader pass | re-solve pass | uplift (reader ✗ → re-solve ✓) | loss (reader ✓ → re-solve ✗) | McNemar p |
|---|---|---|---|---|---|---|
| NLA@42 | Qwen | 187 | 227 | 46 | 6 | 1e-08 |
| NLA@42 | Gemini | 187 | 240 | 62 | 9 | 7e-11 |
| OL@42 | Qwen | 159 | 211 | 59 | 7 | 2e-11 |
| OL@42 | Gemini | 159 | 224 | 82 | 17 | 2e-11 |

Where the uplift comes from:
- **multilingual:** NLA 42 → 61/63, OL 28 → 61/59.
  - For the NLA it is the *concept* unit (47 → 66–71 of 100). The NLA already names the language in 92/100.
  - For OL it is both units: language 56 → 84–90 and concept 40 → 67–68. The re-solve prompt explicitly asks for the language, which inflates OL's uplift on that unit.
- **multihop:** NLA 15 → 26/32. The NLA's readout reconstructs the *problem* well enough for a re-solver to produce the bridge in 11–19 more items than the NLA names itself.
- **basic_readout:** NLA 57 → 65/68.
- **typo:** ≈ no change.

The readouts carry more text-recoverable information than they score with. If the same content were not decodable at all, these items could not be re-solved from the readout alone. Re-solving success is thus an upper bound that the readers' own scores sit under. That is consistent with re-solving fully accounting for regex success, though necessary rather than sufficient evidence for it.

## 4. Matched-layer (L42) comparison

| comparison | typo | multilingual | multihop | basic | pooled (A-only / B-only, p) |
|---|---|---|---|---|---|
| NLA@42 vs OL@42 | 73 v 76 (p=0.61) | 42 v 28 (**p=0.009**) | 15 v 10 (p=0.30) | 57 v 45 (**p=0.023**) | 187 v 159 (54/26, **p=0.002**) |
| NLA@42 vs T_qwen | 73 v 45 (p<1e-4) | 42 v 40 | 15 v 44 (p<1e-5) | 57 v 53 | 187 v 182 (p=0.75) |
| OL@42 vs T_qwen | 76 v 45 (p<1e-5) | 28 v 40 (p=0.09) | 10 v 44 (p<1e-6) | 45 v 53 | 159 v 182 (p=0.09) |
| NLA@42 vs T_gem | 73 v 66 | 42 v 73 | 15 v 75 | 57 v 87 | 187 v 301 (p=2e-18) |

Two things hold at L42:
- The NLA beats OL (multilingual and basic_readout drive it, as in D6/D7).
- Both readers are at or below a text-only model in the NLA's own register, and far below a frontier text model.

## 5. Claim-level text-verifiable precision (100 readouts per reader, 25/family, same items for every reader)

Setup:
- **Judge:** Gemini 3.8 Flash (T=0, reasoning low). It splits each readout into ≤12 atomic claims and labels each with the first matching label of the 5-way codebook in `claims.py`: contradicted > restatement > target/latent-consistent > inferable > unsupported.
- **Judge inputs:** the text, the gold answer and the gold intermediates.
- **TVP** = (restate + inferable + target/latent) / (those + contradicted). Unsupported ≠ false.

| reader | claims | claims/readout | contradicted | restate | target/latent | inferable | unsupported | **TVP [95% CI]** | readouts with ≥1 contradicted |
|---|---|---|---|---|---|---|---|---|---|
| NLA@42 | 844 | 8.4 | 0.36 | 0.09 | 0.11 | 0.38 | 0.06 | **0.62 [0.59,0.65]** | 97/100 |
| OL@42 | 675 | 6.8 | 0.35 | 0.06 | 0.18 | 0.14 | 0.28 | **0.52 [0.46,0.59]** | 77/100 |
| TextNLA (T_qwen s0) | 587 | 5.9 | 0.08 | 0.23 | 0.18 | 0.49 | 0.02 | **0.92 [0.89,0.94]** | 34/100 |

TVP by family (typo / multilingual / multihop / basic):
- NLA: 0.58 / 0.68 / 0.59 / 0.62
- OL: 0.84 / 0.54 / 0.39 / 0.40
- TextNLA: 0.89 / 0.91 / 0.92 / 0.95

**Error profiles differ.**
- **NLA:** its contradictions are mostly *misquotations of the prompt*. 246/301 contradicted claims are quote-like by a crude heuristic, e.g. "The text states: 'Before relegation, the club was forced into eventual delegation'". The NLA reconstructs the prompt nearly but not exactly.
- **OL:** its contradictions are wrong entities and facts, e.g. "The highest summit of the country is Mont Pico". It also has 28% unsupported specifics.
- **Target/latent-consistent claims:** NLA 11%, OL 18%, TextNLA 18%. A text-only model makes as many gold-consistent claims as either reader.

**Judge agreement (40 claims, stratified by reader × Gemini label, re-labelled by Fireworks `accounts/fireworks/models/kimi-k3`):**
- 5-way: raw 0.78, **κ = 0.72** [0.56, 0.87].
- 3-way (text-true / contradicted / unsupported): raw 0.88, **κ = 0.77** [0.57, 0.94].
- The confusions are mostly inferable ↔ target/latent and restate ↔ target, which leave TVP unchanged. Full confusion matrix in `data/claims_tables.md`.

Note this is judge–judge agreement. The prereg's 300-claim human labelling (§6.2) has not been done.

## Caveats

- **Sample-count asymmetry.** Readers are k=1. T_qwen s0 is the only k-matched text arm. T_qwen_k3, T_any and the item-level union are best-of-N and favour text, so read the k=1 rows as primary. Even k=1 ties (NLA) or beats (OL, p=0.09) the readers pooled.
- **Mention scoring.** Regex credits a target mention in any role (pilot weakness #5). TextNLA mentions are subject to the same leniency.
- **Length.** T_qwen outputs are shorter than the NLA's (median 425 vs ~800 chars; the NLA instruction asks for 80–100 words). T_wsb is longer (~1060). The regex's length sensitivity cuts both ways and is covered by the template range.
- **Resolve template.** It explicitly asks for the language and for a correctly spelled entity. That is family-neutral, but it helps multilingual's language unit (mainly for OL).
- **The judge sees gold.** The claim judge sees the gold target and intermediates. "Target/latent-consistent" is gold-referenced, and "unsupported" means only that neither the text nor the gold supports it.
- **Batched-generation numerics.** Left-padded batched greedy generation matched batch-1 for the first 48 tokens on 3/4 prompts. The 4th diverged late, as bf16 batch numerics, the same as 01's check.
- **What "text-explicable" means.** It means the target is recoverable from text, not that the reader used the text. Separating those needs the text-fixed steering arm (H2/H3) or FollowText clamps.

## Costs

- **OpenRouter:** $1.81, all Gemini 3.8 Flash. 1,608 TextNLA generations = $1.34; 300 claim-judge calls = $0.46. Reported `usage.cost`, consistent with the key's usage delta.
- **Fireworks:** 40 kimi-k3 calls, ≤ $0.09 at a conservative price estimate (Fireworks reports tokens only).
- **Total:** ≈ $1.9 of the $40 cap.
- **Modal GPU (estimate):**
  - main run: 6× H200 × 3–6 min (1,600 prompts × k=3);
  - plus an H100 smoke (4 min);
  - plus an aborted H100 run (6 shards, OOM on long prompts, fixed with token-budget batching on H200).
  - About 1.2 GPU-h, ~$6.

## Files (`experiments/06_resolving_decomposition/`)

Code:
- `common.py`: item and readout loading, `WSB_PROMPT_ONLY`, `RESOLVE_TEMPLATE`.
- `env.py`: Modal image (03 pins plus expandable segments).
- `qwen_modal.py`: base-Qwen TextNLA, `.map` over 6 H200 shards, pad check. `::instruction` fetches the NLA instruction.
- `build_jobs.py` → `data/jobs.json`: 1,600 user turns, shared by both backends.
- `llm.py`: OpenRouter/Fireworks client (keys from `.env`, never printed).
- `gemini_gen.py` → `data/gemini_gen.jsonl`.
- `score.py`: writes `judge_in/<arm>/`, runs `wsbench run` → `judged/<arm>/`, collects `data/item_pass.json`.
- `analyze.py` → `data/tables.md`, `data/tables.json`: all decomposition tables, including the per-family McNemar rows and hard reader-only item ids.
- `claims.py` (`judge` | `validate`) → `data/claims_gemini.jsonl`, `data/claims_kimi.jsonl`.
- `claims_analyze.py` → `data/claims_tables.md`, `data/claims_summary.json`.

Data and logs:
- `data/qwen_gen.json`: raw and stripped samples.
- `data/nla_instruction.txt`.
- `data/spend_log.jsonl`, `data/*.log`.

Re-run:

    cd experiments/06_resolving_decomposition
    PYTHONPATH=../../infra modal run qwen_modal.py::instruction && python build_jobs.py
    PYTHONPATH=../../infra modal run --detach qwen_modal.py::main
    ../../.venv/bin/python gemini_gen.py && ../../.venv/bin/python score.py && ../../.venv/bin/python analyze.py
    ../../.venv/bin/python claims.py judge && ../../.venv/bin/python claims.py validate && ../../.venv/bin/python claims_analyze.py

## Prompt-only floor verification (follow-up)

**Verdict: confirmed issue, cause inconclusive.** Under the thinking-off chat rendering that every reader and every bank uses (`docs/read_sites.md`), the frozen floor is about 0.14 too low. Thinking-on and raw rendering both come closer (mean |Δ| 0.08–0.09), but neither reproduces it. The generation code that would settle it is not public.

**Provenance**, from the upstream repo `camilablank/workspace-bench` via `gh api`:
- `prompt_only.json` history: d9a1aa64 (09-16, the lucky-guessing commit), 67cc27d7 (09-18, the 11 new floors), a49d4b04 (09-18), af8059f0 (09-23, mt families re-scored with regex).
- **Generation date.** The six mt entries come from one generation run, `gen-prompt-only-hard-mt`. It was frozen on 2026-09-16 under the MC judge (typo 0.46 / multilingual 0.37 / multihop 0.52 / basic 0.40). On 09-23 it was re-scored with regex "from the SAME prompt-only summaries" (baselines README), giving 0.67 / 0.42 / 0.69 / 0.59. The rows are replicated per layer and judged at L20.
- **Recorded settings.** `_source` records the template and sampling: k=1, T=1.0, top_p 0.95, top_k 64, max_new 256, no stop, empty output_prefix, model `Qwen/Qwen3.6-27B`, no adapter.
- **Not recorded:** chat rendering, the `enable_thinking` setting, the model revision, the seed and the serving stack.
- **The generator is not in the bench.** The README says it is the source repo's `prompt_only_summary` (cf. `olens_sglang`, which suggests SGLang serving). GitHub code search finds no public copy, and the summaries themselves are not published, so I could not check them for `<think>` remnants.
- **Empty cells:** the frozen run has typo 3, multilingual 4, multihop 1, basic 1. None of our 3×400 generations is empty.
- Our model is the volume snapshot `6a9e13bd`.

**Re-run of the frozen template.** Base Qwen, k=1, seed 0, the `_source` sampling, one H100, about 12 GPU-minutes. Shown as pass rate, then (empty cells / `</think>` closed):

| rendering | typo_mt | multilingual_mt | multihop_mt | basic_readout_mt | mean abs Δ vs frozen |
|---|---|---|---|---|---|
| **frozen floor** (empty cells) | 0.67 (3) | 0.42 (4) | 0.69 (1) | 0.59 (1) | – |
| chat, thinking off | 0.83 (0/0) | 0.47 (0/0) | 0.84 (0/0) | 0.80 (0/0) | 0.143 |
| (earlier draw: T_wsb s0/s1/s2) | 0.85/0.83/0.83 | 0.58/0.55/0.51 | 0.82/0.79/0.88 | 0.77/0.78/0.80 | 0.13–0.15 |
| chat, thinking on, thinking text kept | 0.64 (0/0) | 0.59 (0/0) | 0.78 (0/0) | 0.54 (0/0) | 0.085 |
| chat, thinking on, answer after `</think>` only | 0.00 (100/0) | 0.00 (100/0) | 0.00 (100/0) | 0.00 (100/0) | 0.593 |
| raw text, no chat template | 0.80 (0/50) | 0.45 (0/53) | 0.79 (0/41) | 0.65 (0/53) | 0.080 |

The two-sample z-scores against frozen, per family (typo / multilingual / multihop / basic), point the same way:

| rendering | typo | multilingual | multihop | basic |
|---|---|---|---|---|
| thinking off (both draws) | +2.7 to +3.0 | +0.7 / +2.3 | +2.2 to +2.5 | +2.8 to +3.3 |
| thinking on | −0.4 | +2.4 | +1.4 | −0.7 |
| raw | +2.1 | +0.4 | +1.6 | +0.9 |

Reading:
- **Thinking off is significantly above frozen** on typo, multihop and basic, in two independent draws.
- **Thinking on with the thinking text kept** matches typo and basic. Within 256 tokens Qwen never closes `</think>`, so 100% of cells are thinking text, and the regex strips only the tags. But multilingual is too high (+0.17).
- **Thinking on, answer-only** scores 0 everywhere, so the frozen run did not discard thinking content.
- **Raw rendering** is the closest overall but is still +0.13 on typo. In about half its cells the model emits its own empty `<think></think>`.
- **No rendering produces empty cells**, so the frozen run's 1–4 empty cells per family are unexplained. One possibility is a serving-side difference such as SGLang stop or parse behaviour.
- Most cells hit the 256-token cap in every rendering (84–100%), matching the README's truncation note.

**Implications:**
- Read the frozen mt prompt-only floors as a *lower bound* on prompt-only performance.
- With thinking off, the same text-only model with the same template scores about 0.80 / 0.47–0.58 / 0.82–0.88 / 0.77–0.80. That makes the "readers vs prompt-only" gap worse for every reader than the published floors imply.
- **None of the §1–§5 conclusions above depend on the frozen floor.** They use our own text arms.

Files: `floor_modal.py` (3 renderings; `data/floor_gen.json`), `floor_score.py` (`judged/floor_*`, `data/floor_tables.md`).
