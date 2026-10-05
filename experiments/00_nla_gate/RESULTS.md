# 00_nla_gate: does the NLA work, and what does it read? (K0 gate)

Date 2026-10-05. Model: Qwen3.6-27B. Reader: `ceselder/qwen3.6-27b-nla-rl` (snapshot `5a13b7e`).
All numbers come from Modal runs on the 64 activations in `data/example_activations.parquet`
(FineFineWeb text, the activation taken at the last token of `detokenized_text_truncated`).
Raw outputs are in `results/*.json` and on the `mats-results` volume under `/results/00_nla_gate/`.

## K0 verdict: layer convention

**The NLA reads the output of decoder block 42: `model.layers[42]` output = HF
`output_hidden_states[43]`.** That is the same convention as WorkspaceBench's "layer 42", so the
off-by-one feared in `notes/wsbench_internals.md` §0.5 does not exist.

Recompute (`layer_check.py`, H100, bf16, one sequence per forward, `add_special_tokens=True`):
our token counts equal the parquet's `n_raw_tokens` for 64/64 examples.

| candidate | cosine mean / median / min | rel. L2 mean / median / max |
|---|---|---|
| `hidden_states[41]` (block 40 out) | 0.950 / 0.960 / 0.848 | 0.304 / 0.282 / 0.535 |
| `hidden_states[42]` (block 41 out) | 0.974 / 0.979 / 0.914 | 0.222 / 0.205 / 0.406 |
| **`hidden_states[43]` (block 42 out)** | **0.9992 / 0.9999 / 0.975** | **0.026 / 0.017 / 0.233** |
| `hidden_states[44]` (block 43 out) | 0.974 / 0.979 / 0.923 | 0.232 / 0.217 / 0.408 |

- Block 42 output gives the smallest error on **64/64** examples. For every one of them the
  best-matching position is the last token (offset 0).
- Our hooks on blocks 41 and 42 equal `hidden_states[42]` and `[43]` bit for bit (max abs diff
  0.0). This is the WorkspaceBench `Backend.capture` path.
- On cosine: the 0.974 figure belongs to the *wrong* convention (block 41). The correct
  convention reaches a median of 0.9999, so there is no 0.974 cap to explain. Adjacent L42
  residuals all have cosine about 0.97, so relative L2 is the measure that separates them
  (1.7% against 20%).
- Residual mismatch: 6/64 examples have rel L2 > 0.05 (worst is #14 at 0.23). An fp32 capture
  on H200 (`fp32_check.py`) gives the same numbers (#14: 0.216 in fp32, 0.233 in bf16; typical
  examples about 0.007-0.02). So the mismatch is **not** bf16 or kernel noise, and installing
  causal-conv1d would not change it. The likely cause is context or tokenization drift between
  the original extraction (the full document, batched) and re-tokenizing the truncated
  detokenized text. Unresolved, and it does not affect the verdict.
- Code agrees: EasyNLA `HFExtractor` hooks `model.model.layers[K]` and says in its docstring
  that this equals `hidden_states[K+1]`. `NLACriticModel` keeps K+1 = 43 blocks, and the
  shipped AR config has `num_hidden_layers: 43`.
- Functional test (`offbyone.py`, iter_000400, k=4, each vector scored against itself with its
  own predict-the-mean baseline): **FVE 0.760 on our block-42 capture against 0.695 on our
  block-41 capture.** The convention with the higher FVE wins.
  - Caveat: explanations of block-41 vectors still reconstruct the *gold* block-42 vectors at
    0.763. The AR always predicts in block-42 space, and adjacent layers carry nearly the same
    content.
  - So an off-by-one would cost only about 6 FVE points, and is invisible if FVE is scored
    against block-42 targets.
- WorkspaceBench: the public `nla` producer reads `Backend.capture`, which is a forward hook on
  `model.layers[42]`. That is block 42 output, the correct convention.
  - The in-house stored `nla-rl-L42` numbers came from in-house Modal runs whose capture code
    is not public. The docs state every bank is block-output.
  - The lead's rerun with this producer (typo_mt 0.73 against 0.75 stored, multilingual_mt 0.42
    against 0.44) is consistent with that.

## NLA FVE reproduction (`fve_gate.py`, H200; AV = av_base + adapter, AR = shipped `ar_reconstructor`)

FVE follows the EasyNLA "paper" definition:
- Both vectors are rescaled to norm √5120.
- mse_i = the per-element MSE.
- baseline = the per-element variance of the rescaled gold set around its mean. It is computed
  on these 64 vectors and equals 0.4616. The training eval used 128 held-out rows.
- FVE = 1 − mean(mse) / baseline. Failed `<explanation>` extractions are excluded, as in
  training eval.

Settings: k=4 per activation; T=1.0, top_p=1, top_k=0, 256 new tokens (the training eval
settings in run_config); prompt `enable_thinking=False`.

| condition | FVE (mean over samples) | SD across examples | failed / 256 |
|---|---|---|---|
| **iter_000400** | **0.761** (card: 0.766) | 0.176 | 0 |
| iter_000300 | 0.759 (card: 0.756) | 0.168 | 1 |
| av_base, no RL adapter | 0.459 (card step 0: 0.530, measured with that step's AR) | 0.300 | 1 |
| iter_000400, `enable_thinking=True` (EasyNLA default renders an open `<think>`) | 0.754 on valid samples only | — | **205** |
| iter_000400, WorkspaceBench sampling (top_p 0.95, top_k 64) | 0.772 | 0.139 | 0 |
| **controls** | | | |
| shuffled activation (explanation of act[π(i)] scored against act[i]) | **−0.903** | 0.437 | 0 |
| TextNLA: base Qwen3.6-27B writes the explanation from the source text with Sonnet's warm-start instruction prompt, no activation (k=4) | **0.276** | 0.347 | 255/256 closed tags |
| TextNLA, shuffled | −0.797 | | |
| source text itself as the "explanation" (last 64 / 256 / 900 tokens) | 0.151 / 0.198 / 0.208 | 0.44-0.46 | |
| source tail 256 tokens, shuffled | −0.789 | | |
| empty explanation | −0.866 | | |
| constant: the set mean, rescaled to √d | −0.154 | | |

Reading:
- The NLA works as published. iter_000400 reproduces 76.1% against 76.6%, and iter_000300
  reproduces 75.9% against 75.6%, using the final (step-600) AR. The card warns that per-step
  numbers used the AR as it was at that step.
- The non-thinking prompt is the trained contract. With `<think>` left open, 80% of samples
  never close `</explanation>` within 256 tokens.
- WorkspaceBench sampling scores slightly higher (0.772) than full T=1 sampling.
- Text-only controls fall far below the NLA: source text 0.15-0.21, TextNLA 0.28. They are
  well above shuffled (about −0.8 to −0.9), so text-only content explains a real but minority
  share of the variance.
- Important caveats for interpreting "0.28 vs 0.76" as non-text content:
  - (a) The AR was co-trained on the RL AV's register. Qwen-written TextNLA prose is
    off-register for it, and raw source text even more so.
  - (b) The base AV (warm-start, trained to imitate Sonnet's text-only summaries) gets 0.459
    under this AR.
  - So the "prompt re-description" ceiling under a register-matched AR is unknown. It lies
    somewhere between 0.28 and roughly 0.46.
  - A Sonnet-written TextNLA (the actual warm-start generator) would be the fair version.
    That needs an API call, which we deferred.

## FVE as an OOD detector (natural distribution; for pre-registering a margin)

From `score_gate_natural_scoring.json`, iter_000400, natural activations, baseline 0.4616:

| sampling | mean | SD of samples | SD of per-example means | mean within-example SD (k=4) | 5th pct sample | min example |
|---|---|---|---|---|---|---|
| T=1 full | 0.761 | 0.179 | 0.176 | **0.027** | 0.498 | −0.37 (#39) |
| WorkspaceBench sampling | 0.772 | 0.148 | 0.139 | 0.028 | 0.490 | 0.04 |

- Median per-example FVE is 0.813.
- Example #39 is an outlier: a film-listing page whose explanations confabulate a "World War Z"
  plot. The next-worst examples score 0.44-0.46.
- Recommendation: most of the variance is *between* examples. The k=4 sampling noise within an
  example is small (SD about 0.03). So steered-versus-natural OOD tests should be **paired per
  example**: ΔFVE = FVE(steered_i) − FVE(natural_i), averaged over k≥4.
  - The SD of the per-example mean over k=4 is about 0.03/√4 ≈ 0.015, so ΔFVE ≈ 0.04 is the
    smallest shift distinguishable from sampling noise.
  - An unpaired threshold would need about 0.35 (2 between-example SD).
- Pass the natural-set baseline (0.4616, or one recomputed on the natural set of the
  experiment) to `fve_on(..., baseline=...)` when scoring steered vectors.

## Teacher-forced scoring sanity check (`score_gate.py`, iter_000400)

- M[a, w] = log p(candidate_w | AV prompt with act_a injected, right after
  `"<explanation>\n"`), over all 64×64 (activation, item) pairs.
- Candidates:
  - **bare topic word**: the most frequent non-stopword of 5 or more letters in the last 600
    characters of the source, distinct across items.
  - **frame_topic**: `"The text discusses {word}."`
  - **frame_final_token**: `'Final token "{last word of source}"'`, the AV's native phrasing.

| candidate | raw forced-choice acc (own string vs other items' strings, same act) | string-controlled acc (same string, own act vs other acts) | paired log-odds AUROC P(lo_i(c_i vs c_j) > lo_j(c_i vs c_j)) |
|---|---|---|---|
| bare topic word (64) | 0.738 | 0.813 | 0.915 |
| frame_topic (64) | 0.753 | 0.714 | 0.887 |
| frame_final_token (64) | 0.969 | 0.996 | **0.9995** |
| first 10 only: bare / frame_topic / final_token | 0.78 / 0.83 / 1.00 | 0.84 / 0.84 / 1.00 | 0.91 / 0.96 / 1.00 |

- The AV clearly assigns higher log-prob to its own source context's content. The paired
  log-odds design removes the string-prior confound and gives the cleanest signal.
- Caveat: these are in-context lexical features (the last word, a topic word), which also
  appear in the text. Top marks here show readout of *text identity*, not non-workspace
  content.
- Numerical noise: the same scores computed with different batch composition differ by up to
  0.26 nats (bf16). `paired_logodds` against the matrix gave 22.98 against 22.93. Keep both
  candidates of a pair in the same batch (`paired_logodds` does) and treat |Δ| < 0.3 nats as a
  tie.

## Module: `src/mats/readers/nla.py`

- `load_nla(adapter=..., load_ar=True, enable_thinking=False)`: the adapter may be a path, a
  list of paths (switch with `nla.set_adapter("iter_000300")`), or None. Use `nla.no_adapter()`
  for av_base alone.
- `nla.verbalize(acts, k, Sampling(...))` returns `list[list[str|None]]`.
  `verbalize_raw` returns the raw completions.
- `nla.reconstruct(texts)` returns `[N, 5120]`.
- `fve(pred, gold, baseline)`, `predict_mean_baseline(acts)`, `recon_mse`.
- `nla.fve_on(acts, k, baseline=None | float, explanations=None)` gives per-example and
  per-sample FVE and works on arbitrary (steered) vectors.
- `nla.logprob_of_strings(act, cands, prefix="<explanation>\n")` and
  `nla.paired_logodds(act, a, b)`.
- The AR load fails loudly on any missing or unexpected weight except the intentionally
  stripped `model.norm` and `lm_head`.
- AV + AR take about 90 GB, so use an H200. Generation of 256 rows × 256 tokens takes about
  160-240 s (causal-conv1d is not installed, so its torch fallback is used).
- The Modal image is `experiments/00_nla_gate/env.py`: infra pins plus pyarrow, with `src/mats`
  shipped in.

## Oracle Lens contract (`agu18dec/olens_and_ar/olens_s3d_rl600`)

- `run.json` gives:
  - run `s3d.rl.ddp600.s0`
  - transform `unit`, alpha 16000
  - layers **[20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]**
  - prompt `concepts_raw`
  - init from `ao.iolens.ptag.s3d.omp4bp.s2/step251` (GRPO 600 steps, β 0.02, joint bullet-FVE
    reward on the ptag AR)
  - held-out joint FVE 0.487, measured on its own whitened AR. This is not comparable to NLA FVE.
- `README_original.md` is an empty PEFT template. The contract is in the repo-root README.
- **The layer appears in the prompt as a plain integer:** one user turn, chat-templated with
  `add_generation_prompt=True` and `enable_thinking=False`. The prompt reads: `An activation
  vector from layer {layer} of a language model is enclosed in activation tags:
  <activation>{char}</activation>. Produce distinct concepts that encode this activation, each
  as a '- ' bullet on its own line.`
- Injection: the marker ㈜ (id 158983) row of the *input embedding* is **replaced** by
  16000·h/‖h‖. This differs from the NLA, which adds at block 1's output.
- Activations are captured with the adapter disabled, using block-output recorders
  (`ActivationRecorder(blocks, at=LAYERS)`). That is the same block-output convention.
- "Layers outside 20-60 step 4 are off-contract." **L42 is off-contract** for the Oracle Lens,
  both as a vector and as a `{layer}` value never seen in training.
  - For like-for-like comparison with the NLA at L42, also run the Oracle Lens at 40 and 44
    with matching `{layer}`.
- The matched AR (`ar_ptag_pooled`) uses a `[Layer NN]` prompt tag (zero-padded). Its frozen
  `tag_ids` cover the 11 band layers plus L63.

## Decisions / for the lead

1. Use **block-42 output** (`model.layers[42]` forward hook, equivalently `hidden_states[43]`)
   for the NLA everywhere. WorkspaceBench's producer is already correct.
2. Default the AV to **iter_000400, non-thinking prompt**. Use T=1 / top_p=1 / top_k=0 for FVE
   comparisons with the card, and WorkspaceBench sampling for bench comparability; they differ
   by about 0.01 FVE.
3. Run OOD checks with FVE paired per example (k≥4, natural baseline 0.4616 or recomputed).
4. The text-only control is not yet fair. The FVE gap between TextNLA (0.28) and the NLA (0.76)
   partly measures how far TextNLA's register is from the AR's.
   - Before claiming "the NLA's text carries more than re-description", generate a
     register-matched text-only control.
   - Option (a): Sonnet with the warm-start prompt.
   - Option (b): SFT-only text-to-explanation.
5. Not done: the causal-conv1d kernel. Not needed for correctness (an fp32 reference shows the
   same residuals), only for speed.
