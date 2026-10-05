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

## Re-solver consistency check (follow-up)

**The question.** The readout-only re-solver (0.57–0.60) beats the sample-matched text arm T_qwen (0.46). Does the readout carry information the text does not?

**Answer: no. The apparent inconsistency is an instruction effect.**
- T_qwen uses the NLA register, which never asks the model to *name* the answer.
- The re-solver template explicitly asks for the entity, spelled in full, plus the language. That is exactly what the regex rewards.
- With that same template and k=1, a re-solver given the **prompt** scores **0.89**, against **0.60 / 0.56** given only the NLA / OL readout.
- Adding the readout to the prompt does not help: **0.88 / 0.88 vs 0.89**.
- (CPU analysis plus one Gemini run: 1,200 calls, $0.79. Code: `resolve_plus.py`, `data/resolve_plus.jsonl`, `data/resolver_check.json`.)

### (3) What each arm saw

All arms are scored by the same regex instrument at L42. Sampling everywhere: T=1, top_p 0.95 / top_k 64 (Qwen), 256 new tokens (Gemini: max_tokens 256, reasoning minimal), thinking off.

| arm | input | template (exact text in code) | model | k |
|---|---|---|---|---|
| NLA@42 / OL@42 readers | the L42 activation (plus each reader's own carrier prompt) | – | NLA iter400 / OL rl600 | 1 |
| qwen_prompt_nla_s{0,1,2} (T_qwen) | item prompt | NLA warm-start instruction (`data/nla_instruction.txt`): "identify the 2–3 most important features…, ~80–100 words" | base Qwen3.6-27B | 3 (s0 primary) |
| qwen_prompt_wsb_s{0,1,2} (T_wsb) | item prompt + final-token string | wsbench prompt-only template (`WSB_PROMPT_ONLY`): "What would a language model be thinking about on this token?" | base Qwen | 3 |
| gem_prompt_nla (T_gem) | item prompt | NLA instruction | Gemini 3.8 Flash | 1 |
| gem_prompt_wsb | item prompt + final token | wsbench template | Gemini | 1 |
| qwen/gem_resolve_{nla,ol42} | **only** the reader's L42 readout (no prompt) | `RESOLVE_TEMPLATE`: reconstruct the text, **name the entity in full, spelled correctly**, state the language | Qwen (k=3) / Gemini (k=1) | 3 / 1 |
| **gem_rsP** (new, matched control) | **only** the item prompt | `resolve_plus.py`, the same three instructions as `RESOLVE_TEMPLATE` | Gemini | 1 |
| **gem_rsP_nla / gem_rsP_ol42** (new) | item prompt **+** reader readout | the same, with the description inserted after the text | Gemini | 1 |

### (1) Readout-only re-solver hits that every prompt-based arm misses

Prompt-based arms counted here: all 8 earlier samples (Qwen NLA-register ×3, Qwen wsbench ×3, Gemini NLA-register, Gemini wsbench), plus gem_rsP.

| re-solver | hits | not hit by any of the 8 earlier prompt arms | also not hit by gem_rsP |
|---|---|---|---|
| Qwen Resolve(NLA) s0 / s1 / s2 | 227 / 229 / 225 | 1 / 2 / 3 | 0 / 1 / 2 |
| Gemini Resolve(NLA) | 240 | 2 | 1 |
| Qwen Resolve(OL@42) s0 / s1 / s2 | 211 / 216 / 215 | 3 / 4 / 3 | 2 / 2 / 1 |
| Gemini Resolve(OL@42) | 224 | 2 | 0 |
| **union over each reader's 4 re-solver samples** | | NLA 6, OL 4 | **NLA 4, OL 2** |

I inspected all 10 items (6 NLA + 4 OL, before gem_rsP). **Every one is a regex-form artefact, not latent content.** The prompt arms identify the same concept, but in a form the multi-token-only contract does not credit.

- **Uncredited English short forms.** Prompt arms say "ladybug" (credited forms: "ladybird beetle" / "lady beetle"), "syringe" ("hypodermic/medical syringe"), "snail" ("garden/land snail"), "iron" ("clothes iron") and "hail" / "hujan es" ("hailstone"). The prompt arms also name the concept in the target language, which satisfies only the language unit; e.g. gem_rsP writes "strzykawka (Polish for syringe)" and "salyangoz (meaning snail)".
  - These items: biedronka, strzykawka, salyangoz, esokabat, clothesiron (NLA); hailstone, strzykawka, clothesiron, silitysrauta (OL).
- **OL's Chinese code-switching.** OL readouts contain 冰雹, 注射器 and 熨斗, which are credited `zh` forms. The Qwen re-solver copies them into its output, and Qwen's OL hits on hailstone, strzykawka and clothesiron are all via `zh`.
- **Name order.** On multihop charlotte-guest-dowlais, the prompt arms name Lady Charlotte Guest and the Dowlais Ironworks but write "John Josiah Guest", while the credited form is "Josiah John Guest". The Gemini readout re-solver wrote the credited order.

Fine-grained per-item hit units are in the inspection output above, reproducible from `judged/*/multilingual_mt/results.json`.

### (2) k-matched comparisons (each prompt arm k=1 vs each re-solver sample k=1)

Each pair below is (re-solver-only items, prompt-only items).

| re-solver (k=1) | vs T_qwen (NLA register) | vs T_wsb s0 | vs T_gem | vs gem_wsb | vs **gem_rsP (same template)** | McNemar p vs gem_rsP |
|---|---|---|---|---|---|---|
| Gemini Resolve(NLA) | 107 / 49 | 42 / 104 | 44 / 105 | 78 / 88 | **7 / 124** | 9e-29 |
| Gemini Resolve(OL@42) | 106 / 64 | 42 / 120 | 42 / 119 | 80 / 106 | **7 / 140** | 3e-33 |
| Qwen Resolve(NLA) s0 | 104 / 59 | 43 / 118 | 42 / 116 | 76 / 99 | – | – |
| Qwen Resolve(OL@42) s0 | 98 / 69 | 32 / 123 | 39 / 129 | 73 / 106 | – | – |

- **Against T_qwen only, the re-solver looks stronger** (about 105 vs 55 one-sided items). That is the inconsistency Fable flagged. It disappears whenever the prompt arm is also asked to name things.
- **The fair pair is same model, same template, same k.** Gemini prompt-only re-solver 0.89 [typo .90 / multilingual .83 / multihop .93 / basic .91] vs readout-only 0.60 (NLA) / 0.56 (OL): 124–140 items where the prompt wins, against 7 where the readout wins.

### (4) Readout + prompt (second-stage extraction)

| Gemini re-solver, same template | typo | multilingual | multihop | basic | pooled |
|---|---|---|---|---|---|
| prompt only (gem_rsP) | 0.90 | 0.83 | 0.93 | 0.91 | **0.89** |
| prompt + NLA@42 readout | 0.91 | 0.84 | 0.88 | 0.88 | **0.88** (13 gained / 19 lost, p=0.38) |
| prompt + OL@42 readout | 0.89 | 0.83 | 0.92 | 0.88 | **0.88** (13 / 18, p=0.47) |
| readout only, NLA / OL (for reference) | 0.77 / 0.73 | 0.63 / 0.59 | 0.32 / 0.28 | 0.68 / 0.64 | 0.60 / 0.56 |
| reader's own regex score, NLA / OL | 0.73 / 0.76 | 0.42 / 0.28 | 0.15 / 0.10 | 0.57 / 0.45 | 0.47 / 0.40 |

**Two layers of underestimation:**
- **The regex underestimates what the readouts contain.** Second-stage extraction from the readout alone adds +0.13 (NLA) and +0.16 (OL) over each reader's own regex score.
- **All of that extracted information is already in the prompt.** Adding the readout to the prompt changes nothing (−0.01, n.s.). On multihop, the NLA readout slightly *hurts* (8 lost vs 3 gained, p=0.23), consistent with its prompt misquotations (§5).

### Verdict and wording we can defend

**"Fully explained by text" holds**, at the item level and under a matched-template, k=1 comparison. The defensible wording:

> On WorkspaceBench's four regex multi-token families at L42 (n=400), the target information in NLA and Oracle Lens readouts is recoverable from the prompt text alone:
> - A text-only LLM given the prompt names the target more often than the same LLM given only the reader's readout (0.89 vs 0.60 / 0.56, same template, k=1).
> - Adding the readout to the prompt does not increase it (0.88 vs 0.89).
> - Of 400 items, 4 (NLA) and 2 (OL) are recovered from the readout but by none of 9 prompt-based samples. On inspection all of these are regex-form artefacts: the prompt-based arms name the same concept in an uncredited form, or OL code-switches into a credited Chinese form.

Do not say:
- **"The readers re-solve from text."** These data show the readouts' scored content is text-recoverable. They cannot show which mechanism produced it.
- **"The readouts contain nothing beyond the text."** We tested only the regex targets of these families, not other content.

Also report: reader regex scores *understate* their readouts' recoverable content by 0.13–0.16 (the second-stage extraction effect). Comparisons between a reader and a text baseline are only fair under a matched instruction: the T_qwen tie (0.47 vs 0.46) depends on the NLA register, which never asks for the answer.

**Costs, updated:** OpenRouter total ≈ $2.60 (+$0.79 for this check), Fireworks ≤ $0.09.
