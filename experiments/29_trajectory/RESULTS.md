# Exp 29: naming and carrying along the NLA's training trajectory

**Question.** Does the NLA's not-naming of structural state (and its naming of the language) come from the RL reward,
or was it fixed already by the warm start? The sprint's thesis says the NLA does not name structural state because its
RL reward runs through a reconstructor that does not pay for statements. The reviewer memo
(`notes/draft_critique_reviewer.md`, hole 1) puts the confound first:

> "Naming follows the training reward" is not separated from "naming follows the SFT data", and the instrument that
> is supposed to separate them is downstream of the same data. The NLA's warm start is Sonnet summaries written from
> the prefix text only [...]; a summary of a Python file says "Python" and never says "inside the third call
> argument". That register alone predicts every naming cell in exps 18, 19, 22 and 23 with no reward story. [...]
> What closes it: naming and carrying rates as a function of RL step, from `av_base` (pre-RL, loadable via
> `nla.no_adapter()`) through `iter_000300`, `iter_000400`; and a text-only TextNLA (exp 06/07 infra) on the same
> bank. If all are flat, the "property of the training reward" clause has to go.

The test: the same naming and carrying measures on the NLA at several points of its training, plus a text-only zero
point, on the exp 22 public bank, with exp 22's own judge, regexes, critic and lines, unchanged.

## Checkpoints

| point | what | where |
|---|---|---|
| `textnla` | base Qwen3.6-27B given the prompt TEXT (the item's token ids decoded) under the NLA warm-start instruction (exp 06's arm, `06_resolving_decomposition/data/nla_instruction.txt` = `nla_meta.yaml` api_summaries.instruction_prompt); chat template, thinking off | H100, stage text |
| `av_base` | the warm-start verbalizer (Qwen3.6-27B with the SFT LoRA merged), no RL adapter: `load_nla` with the adapters loaded and `nla.no_adapter()` (`PeftModel.disable_adapter()`) around generation | H200, stage gen |
| `iter_000300`, `iter_000400` | the two GRPO adapters mirrored on the volume (`rfm-weights`, snapshot `5a13b7ec`, `av_rl_adapters/`); iter_000400 is `nla.DEFAULT_ADAPTER`, the adapter every sprint experiment used | H200, stage gen |
| `iter_000100`, `_000200`, `_000500`, `_000600`, `_000700`, `_000800` | in the HF repo (`ceselder/qwen3.6-27b-nla-rl`, same revision, 1.87 GB each, r64 all-module LoRA, one per 100 GRPO steps; the repo's held-out FVE goes 53% at step 0 to about 78% by step 600); the fetch stage downloads them into the volume snapshot before the sweep | CPU, stage fetch |

The critic is the repo's single final reconstructor (the repo saves no per-step AR), the one every sprint experiment
pairs with iter_000400; it scores every checkpoint, including av_base and textnla. Carrying at a checkpoint is
therefore "carried in a direction the final critic reads", the same measure exp 25 took for iter_000400.

## Pipeline (`run29.sh`; app `m7cx`)

1. `fetch` (CPU): the missing `iter_*` adapters at the pinned revision into the volume's HF cache; nothing deleted.
2. `text` (H100, `s5nt`): TextNLA, k=1 on the selection split (160 items per row), k=8 on the mining split (40 items),
   NLA sampling (T=1, no top-p/top-k, 256 new tokens; the same `Sampling()` the NLA checkpoints use, so the only
   difference along the x-axis is the model) -> `gen29_textnla_{row}.json`.
3. `gen` (H200, `d3wq`, one container per checkpoint): av_base + all adapters loaded once, the active adapter switched
   per checkpoint; k=1 readouts on the selection split and k=8 on the mining split -> `gen29_{ckpt}_{row}.json`;
   then the k=1 readouts through the critic, scored by exp 14's frozen probe against the true layer-42 activations,
   class-direction cosine and FVE -> `recon29_{ckpt}_{row}.json`. The first launch is `--ckpts av_base,textnla`:
   for textnla the function is recon-only (it reads stage text's file), so no extra pass is needed.
4. Laptop: `judge29.py` runs exp 22's states judge (exp 12 `run_judge`/`recall_lib`: Gemini 3.8 Flash, T=0, 20
   readouts per call, the same prompt, schema and row descriptions) on the mining readouts of every checkpoint ->
   `results/judged/{ckpt}_{row}.json`, exp 22's format; `analyze29.py` -> `results/table29.md`, `table29.json`,
   `fig29.png`.

Measures per checkpoint and row: **states** (judge stated rate, exp 22's `nla_states` cell), **names** (strict regex
on the k=1 readouts, exp 23/26's `NAME_RE`), **carries** (critic + frozen probe AUROC, exp 25's measure), **textbag**
(bag-of-tokens AUROC on the readout text, exp 20/23's cross-check), plus FVE as the reward achieved.

## Cost estimate (written before the run)

GPU: text stage about 1 H100-hour; one checkpoint (1,280 k=1 plus 2,560 k=8 generations of 256 tokens, plus the
critic) about 1.5 to 2 H200-hours. Three NLA points (av_base, iter_000300, iter_000400) + textnla: about 7 GPU-hours,
$30 to $40. All nine NLA points (after fetch): about 19 GPU-hours, $80 to $90, sequential, one GPU at a time.
Judge: exp 22's rate ($1.17 for 4,096 readouts) gives $0.73 per checkpoint at 8 samples per item (2,560 readouts, 128
calls), far under the $15 per checkpoint line, so 8 samples per item are kept; ten checkpoints about $7.50.

## Predictions written before the run

1. Structural rows: the judge-scored states rate stays at or under 0.25 at every checkpoint, including av_base and
   TextNLA.
2. code_lang: the states rate rises by at least 0.15 from av_base to iter_000400. If it does not, the reward clause
   is dropped from the thesis and naming is attributed to the warm start.
3. Carrying through the critic on the readable rows (py_string, js_string, latex_math, py_arg, json_depth) rises from
   av_base to iter_000400 by at least 0.05 on average (the reward pays for carrying), and TextNLA's carrying is
   within 0.05 of av_base.
4. Text-bag AUROC on the readouts changes by under 0.05 across checkpoints on the string rows (py_string, js_string,
   latex_math).

`lib29.check_predictions` scores these four from `table29.json`; `analyze29.py` prints them.

## Results

Ran 27 to 28 Oct 2026; iter_000600 was run on 28 Oct and iter_000800 folded in, so every checkpoint from the warm
start to step 800 is scored. Table `results/table29.md` (also `table29.json`), figure `results/fig29.png`, judge
outputs `results/judged/`. Judge spend $1.36 for the last pass. The iter_000400 carrying cells reproduce exp 25 within
0.02 on every row.

Structural rows below means py_string, js_string, latex_math, py_arg, json_depth, md_heading, md_list.

| measure | textnla | av_base | iter_000100 to iter_000800 |
|---|---|---|---|
| states (judge), structural rows | 0.17 to 0.49 (py_string 0.47, js_string 0.49) | 0.01 to 0.21 | 0.00 to 0.29, no trend (py_arg drifts down 0.12 to about 0.03) |
| states (judge), code_lang | 0.50 | 0.55 | 0.71 to 0.77 from step 100 on |
| names (regex), code_lang | 0.42 | 0.46 | 0.62 to 0.75 |
| carries, readable rows | 0.84 to 0.99 | 0.87 to 0.99 | 0.91 to 1.00, flat after step 100 |
| FVE, py_string | -0.05 | 0.18 | 0.53 at 100 rising to 0.59 at 800 |
| cosine to the class-mean direction, py_string | 0.56 | 0.79 | 0.91 to 0.92 |

Predictions:

1. FAILS, on TextNLA: the text-only NLA states string and Markdown structure far more often than any activation
   reader (py_string 0.47, js_string 0.49, md_heading 0.30). Every activation checkpoint stays at or under 0.25 except
   js_string at step 200 (0.29). The prediction's intent, that RL does not make the NLA state structure, holds.
2. Holds: code_lang states rise 0.55 to 0.71 by step 400 (+0.16), and the rise is complete by step 100.
3. FAILS, on the ceiling: carrying on the readable rows rises only +0.02 on average from av_base to step 400 because
   av_base already carries at 0.87 to 0.99. The second half holds: TextNLA is within 0.05 of av_base (-0.04).
4. FAILS narrowly: the text-bag AUROC spans 0.047 on py_string, 0.082 on js_string and 0.057 on latex_math across ten
   checkpoints, with no trend; this is sampling spread around 0.90, not drift.

Reading. The low structural naming is set before RL. The text-only warm start describes the prompt and says where in
a string or heading the reader is; the activation-verbaliser warm start (av_base) already drops most of that, and the
800 RL steps leave it where av_base put it while raising reconstruction a great deal (FVE 0.18 to 0.59 on py_string,
cosine 0.79 to 0.91) and raising code-language naming (judge 0.55 to 0.71-0.77, regex 0.46 to 0.62-0.75), with both changes mostly complete by step 100.
Carrying of the structural state is near ceiling from av_base on. So the reward pays for reconstruction and for the
one property (the language) that the describing register already names, and neither pays for nor removes structural
statements; exp 30 shows the statements can be added at no cost to reconstruction.
