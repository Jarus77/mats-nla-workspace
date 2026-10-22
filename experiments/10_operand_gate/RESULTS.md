# 10 Operand gate — stage 1 (G1/G2/G3), PILOT, EXPLORATORY

**Status:** pilot, exploratory. Prompted by an external result seen 2026-10-12 (M. Kenney,
[nla-consistency](https://github.com/mkenney2/nla-consistency)): on 600 "a ± b =" prompts a linear probe reads
both operands from the layer-42 activation at "=" (exact: a 83%, b 93%) while the NLA reports them 5% / 38%.
Not part of the pre-registered Holm family; any follow-up enters prereg §3.5 as a declared secondary with this
disclosure. G4 (causal use) and NLA tracking are **stage 2**, not run here.

**Question.** Is operand identity at "=" inside our J-space (prereg §4.1: top-r right singular vectors of
W_U·diag(1+w)·J_42, neuronpedia n1000; r* = 384, 4r* = 1536)?

## Data
- `data/arithmetic.jsonl`: his 600 prompts and stored probe/NLA readouts, copied verbatim from
  `https://raw.githubusercontent.com/mkenney2/nla-consistency/main/data/arithmetic.jsonl`
  (sha256 `1dbfd7a47b9a7f265343116779ea766089ad0dd364b8198b2240dff1681aa6bf`).
  a ∈ [2, 99], b ∈ [1, 99], answers ∈ [1, 190], 300 "+" / 300 "−"; a < 10 in 54 items, b < 10 in 130.
  His stored per-digit probe predictions give exact-operand accuracy 0.827 (a) / 0.932 (b), i.e. his 83% / 93%.
- Capture: bare prompt, no chat template, no few-shot, L42 (block-42 output = HF hidden_states[43]) at the
  final token (" ="), bf16 model, one forward pass per prompt. Qwen tokenises digits one per token, so each
  operand is two 10-class slots (tens, units); there is no whole-number token to test.

## Method (`o10_modal_rfm.py`, `o10_lib.py`; 04's `bf_gate_lib` conventions)
- G1: one-vs-rest L2 logistic probes per digit slot, 5-fold CV (one prompt = one group); accuracy = argmax
  (chance 0.10) with a Wilson CI; macro OvR AUROC with an item-bootstrap CI. Exact operand = both digits right.
- G2: enrichment of each unit direction in the J basis, ‖Q_rᵀv‖² / (r/d) (random = 1.0): the 10 one-vs-rest
  class-mean (diff-in-means) directions, the 10 probe-weight directions, a shuffled-label class-mean reference
  (10 permutations × 10 classes) and an isotropic random band (K = 100). Also a ridge "magnitude" direction
  per quantity (a, b, answer), since the NLA keeps rough size (his r = 0.57 / 0.88).
- G3: standard J-Lens readout softmax(W_U·norm(J·h)); class score = −log10(best rank of the digit token + 1);
  stat = mean OvR AUROC vs a 200-permutation null (04 `functional_stat`); plus the rate at which the true digit
  token sits in the lens top-10.
- Label (prereg §4.2, G1–G3 only): IN-J if the class-mean enrichment median > 1.5 at r* or 4r*;
  DIFFUSE-BUT-J-READABLE if ≤ 1.5 but the lens reads it; else NON-J pending G4.

## Results

Run: Modal app `rfm`, function `operand_gate`, call `fc-01M4QMBC4YKVACV79Q60ZTVAEE`, A100-80GB ×1, 2026-10-12.
Function time 380.6 s (model load + capture 111 s; gates 270 s); ≈ 10 min billed GPU including container start,
well under $1 at Modal's A100-80GB rate. Outputs on `rfm-results`: `10_operand_gate/acts_hs43.pt` ([600, 5120]
fp32 at L42, the model's next token, tokenisation facts), `dirs.pt` (class-mean, probe-weight and magnitude
directions), `gate.json`; local copies in `results/` (`acts_hs43.pt` not downloaded).

**Checks.** Tokens of "51 - 49 =": `['5', '1', ' -', ' ', '4', '9', ' =']`; `tok.encode("51")` has 2 tokens, so the
tens/units slots are the whole story. The model's greedy next token on the bare prompt is " ?" (493/600) or " "
(105), 98% agreement with his `model_next`: on a bare prompt the model does not emit the answer at "=".
Random isotropic enrichment band (K = 100): r* 0.99 [0.88, 1.11], 4r* 1.00 [0.95, 1.05]. The activation
variance itself is 1.19× enriched in J at both ranks, and the shuffled-label class-mean reference sits at that
same 1.17–1.19 (a one-vs-rest class mean of a random 10% subset is mostly the grand-mean direction).

| slot | G1 acc (chance 0.10) | G1 macro AUROC | G2 class-mean enrichment r*/4r* median (range) | G2 probe-weight | G2 shuffled-label ref | G3 J-Lens OvR stat [null 95%] | true digit in lens top-10 | label |
|---|---|---|---|---|---|---|---|---|
| a_tens | 0.94 [0.92, 0.96] | 0.99 [0.99, 1.00] | 1.16 (1.05-1.23) / 1.14 (1.08-1.20) | 0.71 (0.62-0.76) / 0.79 (0.71-0.80) | 1.19 (0.89-1.49) / 1.19 (1.03-1.37) | 0.523 [0.493, 0.506] | 0.00 | **DIFFUSE-BUT-J-READABLE** |
| a_units | 0.88 [0.85, 0.90] | 0.97 [0.97, 0.98] | 1.12 (0.96-1.22) / 1.15 (1.06-1.17) | 0.65 (0.62-0.72) / 0.77 (0.74-0.83) | 1.19 (0.94-1.44) / 1.19 (1.04-1.42) | 0.515 [0.494, 0.507] | 0.00 | **DIFFUSE-BUT-J-READABLE** |
| b_tens | 0.97 [0.95, 0.98] | 0.99 [0.99, 1.00] | 1.14 (1.03-1.28) / 1.16 (1.08-1.38) | 0.69 (0.62-0.76) / 0.82 (0.77-0.89) | 1.18 (0.86-1.45) / 1.19 (1.03-1.41) | 0.571 [0.488, 0.512] | 0.00 | **DIFFUSE-BUT-J-READABLE** |
| b_units | 0.98 [0.96, 0.99] | 1.00 [1.00, 1.00] | 1.26 (1.16-1.43) / 1.49 (1.28-1.78) | 0.72 (0.61-0.86) / 0.85 (0.78-0.90) | 1.17 (0.94-1.52) / 1.17 (0.97-1.35) | 0.581 [0.494, 0.506] | 0.00 | **DIFFUSE-BUT-J-READABLE** |
| ans_tens | 0.42 [0.39, 0.46] | 0.79 [0.76, 0.81] | 1.29 (1.07-1.34) / 1.18 (1.03-1.21) | 0.67 (0.61-0.73) / 0.77 (0.72-0.82) | 1.15 (0.94-1.41) / 1.18 (0.99-1.36) | 0.564 [0.488, 0.512] | 0.00 | **DIFFUSE-BUT-J-READABLE** |
| ans_units | 0.35 [0.32, 0.39] | 0.75 [0.73, 0.77] | 1.28 (1.14-1.45) / 1.23 (1.03-1.31) | 0.71 (0.65-0.80) / 0.83 (0.78-0.84) | 1.17 (0.82-1.48) / 1.18 (1.04-1.32) | 0.515 [0.493, 0.506] | 0.00 | **DIFFUSE-BUT-J-READABLE** |

- exact_a: 0.825 [0.793, 0.853]
- exact_b: 0.948 [0.928, 0.963]
- exact_ans: 0.178 [0.150, 0.211]
- op: G1 AUROC 1.00 [1.00, 1.00] acc 1.00; G2 384: 1.27, 1536: 1.18; G3 stat 0.631 null [0.4815978472222222, 0.5163836111111111] readable=True
- magnitude a: CV Pearson r 0.99, MAE 2.7; enrichment 384: 0.66, 1536: 0.82
- magnitude b: CV Pearson r 1.00, MAE 1.7; enrichment 384: 0.69, 1536: 0.81
- magnitude answer: CV Pearson r 0.99, MAE 4.0; enrichment 384: 0.74, 1536: 0.86


- Exact operand (both digits right): **a 0.825 [0.79, 0.85]**, **b 0.948 [0.93, 0.96]** (his 0.827 / 0.932: replicated);
  answer 0.178 [0.15, 0.21] (not computed at "=" on a bare prompt; see Checks).
- Operator: G1 AUROC 1.00; G2 1.27 / 1.18; G3 stat 0.631 [0.48, 0.52] (per class: "−" 0.38, "+" 0.88): the lens
  weakly echoes the "+" token that is in the prompt. DIFFUSE-BUT-J-READABLE.
- Magnitude (ridge on a, b, answer): CV Pearson r 0.99 / 1.00 / 0.99 (MAE 2.7 / 1.7 / 4.0); enrichment
  **0.66 / 0.82**, **0.69 / 0.81**, **0.74 / 0.86** at r* / 4r*: below the random band at r*.
- Probe-weight directions: enrichment 0.65–0.72 at r*, 0.77–0.85 at 4r*, below the random band at r* for every slot.
- G3 detail: the true digit token never reaches the lens top-10 (0/600 in every slot); its median rank is
  816–1446 (vocab ≈ 150k); the lens top-10 at "=" is underscores (`____`, ` ___`, …) and `？` in all 600 items,
  i.e. the model's own next-token guess. Per-class AUROCs are monotone in digit value for the tens slots
  (b_tens 0 → 9: 0.38, 0.44, 0.42, 0.47, 0.54, 0.67, 0.61, 0.67, 0.67, 0.82; ans_tens 0.44 → 0.75), so the weak
  lens signal is a magnitude/recency correlate, not digit identity.

## Reading (pilot; G4 and NLA tracking not run)
1. **G1 replicates his probe result on our capture** (exact a 0.825 vs his 0.827; b 0.948 vs 0.932).
2. **Operand identity is geometrically J-depleted.** Class-mean enrichment (1.12–1.29 at r*, 1.14–1.49 at 4r*) is
   no higher than its own shuffled-label reference (1.17–1.19), the 04 pattern for "uninformative enrichment";
   the directions that carry the information, probe weights (0.65–0.85) and the magnitude regressors
   (0.66–0.86), are *below* random, like col (0.67 / 1.10) rather than tense (5.6 / 2.3) or language (3.3 / 1.7).
3. **The J-Lens does not emit the digits** (never top-10; rank ~10³), but its digit ranking is statistically above
   the permutation band at AUROC 0.52–0.58, carried by a monotone magnitude correlate in the tens slots. By the
   letter of §4.2 that is **DIFFUSE-BUT-J-READABLE** for every slot; by effect size it is far from the
   in-J references (tense lens 0.52 was labelled IN-J only on enrichment; language lens 0.62) and far from col's
   0.500 [0.499, 0.501]. The §4.4.3 leak test (digit tokens as the content set, two-sided vs the J-complement band)
   is the gate that matters for a steering dose, and it has not been run.
4. **The answer is not computed at "=" on bare prompts** (exact 0.18; next token " ?"), so his "the model uses the
   class-mean directions" (28% answer flips) rests on the few-shot framing. Stage 2's G4 must use few-shot items
   where the model answers, as his did; the bare-prompt activations here are fine for G1–G3 but not for G4.

## What this does and does not change
- Does not add a primary. If stage 2 passes (G4 with centred J-removed steering on few-shot items; NLA 8-sample
  tracking vs random bands), operand identity enters prereg §3.5 as a declared secondary with the 2026-10-12
  disclosure. Holm stays over the 4 primaries.
- Candidate wording if it passes: "a second J-depleted, causally used feature (two-digit operand identity at
  '='), decodable by a probe, which the NLA tracks coarsely (magnitude, not digits) and the J-Lens carries only as
  a weak magnitude correlate." Until G4 and the leak test are run, the honest label is
  **geometrically non-J, lens-marginal, G4 pending**.
- Note for future Modal files: `max_inputs=1` is deprecated in this Modal version (warning only; use
  `single_use_containers=True`).

## Files
`env.py`, `o10_lib.py`, `o10_modal_rfm.py`, `spawn_driver.py`, `watch.sh`, `o10_analyze.py`, `data/arithmetic.jsonl`,
`results/{gate.json, dirs.pt, run_summary.json, analysis.md, call_ids.json, watch.log}`.


---

# Stage 2a — few-shot bank, leak test, model-side G4, J++ second basis (PILOT, EXPLORATORY)

**Status:** pilot, exploratory, model-side only (no NLA, no Oracle Lens). Same disclosure as stage 1 (prompted by an
external result seen 2026-10-12). Nothing here is a primary; it informs whether operand identity can enter prereg §3.5 as
a declared secondary (S10) and what the stage-2b reader test would need.

## Design (`o10s2_modal_rfm.py`, `o10s2_lib.py`; tests in `test_o10s2_lib.py`)
- **Few-shot bank:** each of the 600 items is preceded by 3 deterministic exemplars of the same format with correct
  answers (seed 0 per item, operands disjoint from the item's (a, b), positive results), e.g.
  `'89 + 41 = 130\n90 + 85 = 175\n31 + 16 = 47\n51 - 49 ='`; 36–43 tokens. Capture at the final " =" (L42 = hidden_states[43]) and, for the answer
  reference, at an appended " " token (the position whose next token is the first answer digit). The model answers
  **1.000** of the items correctly (greedy; first generated token is " " in 1.000).
- **Directions (2-fold cross-fit by item; steer fold f with directions fitted on the other fold):** `mag_a_raw` = ridge
  regression direction for the first operand's value; `mag_a_jrem` = the same with the top-1536 J-space projected out;
  `cm_atens_jrem` = one-vs-rest class-mean direction for the most frequent tens digit of a (digit 5), J-removed.
  Doses Δ = q90 − median of the natural projection on the training fold (prereg §4.4.1; `mats.capture.doses_from_projection`).
- **Leak (prereg §4.4.3):** centred h' = h − (v·h)v ± Δv on the captured vectors; J-Lens n1000 set score −log10(best rank + 1)
  for a content set (digit tokens) and a label set (number/digit/operand variants); paired +/− AUROC and mean shift against
  K = 100 J-complement directions at the same ‖Δ‖, two-sided 95% band, z reported.
- **G4 (model side):** the same centred edit at "=" in the forward pass (prefill only; KV-cache decode unaffected),
  16 rows per forward; measures: greedy answer flip rate vs the unsteered answer (4 new tokens, parsed integer), signed
  numeric shift of the greedy answer along the expected direction (+Δ on a-magnitude should raise the answer for both ops;
  for the class-mean direction, towards the answer with a's tens digit set to 5), and the first-digit log-odds
  (correct digit vs the best other digit) at the position after " ". Nulls: K = 20 J-complement + 20 isotropic random
  directions at the `mag_a_jrem` dose, 10 + 10 at the `cm_atens_jrem` dose; z vs each band. The raw a-magnitude
  direction is the reference for the retained fraction.
- **J++:** `koayon/jpp-lenses` `qwen3.6-27b/lens.pt` (inspected: `parameters.jacobians[str(layer)]`, [5120, 5120] per
  layer 0–62, config `hf_model_name = Qwen/Qwen3.6-27B`, `lens_type = jacobian`, `checkpoint_name = jpp_trained_lens`,
  64 training prompts); layer-42 basis built as ours (top right singular vectors of W_U·diag(1+w)·J++_42, r_max 1536);
  enrichment of every stage-1 and few-shot direction under both bases.

## Run
Modal app `rfm`, A100-80GB ×1, 2026-10-12. First attempt (call `fc-01M4QNJQ92JDHP0TZ1D2TX9A8Z`) OOM'd at the start of
G4 with a 127-row batch inside fla's `chunk_gated_delta_rule` (79 GB in use) and was cancelled; relaunched
(`fc-01M4QPBWXXS9WX08M3R7GYWPHY`) with per-stage checkpoints on `rfm-results` and G4 chunked to 16 rows per forward.
Function time 2917 s (capture 53 s, G4 2543 s, gates + leak ≈ 322 s); with the
aborted first attempt (≈ 10 min) and container starts, ≈ 70 A100-minutes, roughly $3–4 at Modal's rate; the J++ steps
(CPU download 31 s; A10G enrichment) add well under $1. Outputs on `rfm-results/10_operand_gate/stage2a/`:
`acts_fewshot_hs43.pt` (H at "=" and at the space, [600, 5120] fp32), `dirs_fewshot.pt`, `g4_raw.pt`, `gate_fewshot.json`,
`leak.json`, `stage2a.json`, `jpp.json`; local copies in `results/stage2a/` (activations not downloaded).

## Results

### Few-shot bank: G1–G3 (n = 600, 5-fold CV; enrichment r*/4r*, random = 1.0)

Random band at r*: [0.88, 1.11]; at 4r*: [0.95, 1.05].

| slot | G1 acc | macro AUROC | G2 class-mean (median) | G2 probe-weight | shuffled-label ref | G3 lens stat [null 95%] | true digit in top-10 | label (§4.2 letter) |
|---|---|---|---|---|---|---|---|---|
| a_tens | 0.79 | 0.96 | 1.03 / 1.68 | 0.59 / 0.84 | 1.11 / 1.25 | 0.555 [0.489, 0.510] | 0.02 | **IN-J** |
| a_units | 0.51 | 0.86 | 0.98 / 1.61 | 0.65 / 0.79 | 1.14 / 1.25 | 0.537 [0.489, 0.510] | 0.02 | **IN-J** |
| b_tens | 0.69 | 0.92 | 1.06 / 1.18 | 0.64 / 0.80 | 1.11 / 1.26 | 0.600 [0.486, 0.519] | 0.03 | **DIFFUSE-BUT-J-READABLE** |
| b_units | 0.84 | 0.98 | 1.34 / 1.77 | 0.67 / 0.83 | 1.09 / 1.24 | 0.600 [0.490, 0.513] | 0.02 | **IN-J** |
| ans_tens | 0.34 | 0.73 | 1.62 / 1.38 | 0.60 / 0.71 | 1.09 / 1.22 | 0.556 [0.484, 0.515] | 0.00 | **IN-J** |
| ans_units | 0.27 | 0.66 | 1.31 / 1.23 | 0.63 / 0.72 | 1.12 / 1.25 | 0.516 [0.491, 0.511] | 0.02 | **DIFFUSE-BUT-J-READABLE** |
| ans_tens@space | 0.24 | 0.61 | 1.26 / 1.11 | 0.82 / 0.87 | 1.28 / 1.09 | 0.500 [0.489, 0.512] | 0.01 | **NON-J (G1-G3; G4 pending)** |
| ans_units@space | 0.19 | 0.57 | 1.23 / 1.07 | 0.82 / 0.86 | 1.29 / 1.08 | 0.504 [0.496, 0.506] | 0.01 | **fails G1** |

- exact_a: 0.415 [0.376, 0.455]
- exact_b: 0.612 [0.572, 0.650]
- exact_ans: 0.138 [0.113, 0.168]
- exact_ans@space: 0.097 [0.076, 0.123]
- few-shot model answers correctly (greedy, unsteered): 1.000; first generated token is the space: 1.000
- magnitude a: CV r 0.95, MAE 6.1; enrichment 0.66 / 0.79
- magnitude b: CV r 0.97, MAE 5.1; enrichment 0.67 / 0.80
- magnitude answer: CV r 0.98, MAE 8.1; enrichment 0.77 / 0.83
- G4 directions (cross-fit folds): enrichment mag_a_raw 0.72 / 0.79; mag_a_jrem 0.00 / 0.00; cm_atens_raw 0.90 / 1.69; cm_atens_jrem 0.00 / 0.00
- doses (q90 − median, per fold): {'0': {'mag_a_raw': 2.342, 'mag_a_jrem': 1.67, 'cm_atens_raw': 4.734, 'cm_atens_jrem': 4.442}, '1': {'mag_a_raw': 2.108, 'mag_a_jrem': 1.441, 'cm_atens_raw': 8.506, 'cm_atens_jrem': 7.164}}; ‖Δ‖/median‖h‖: mag_a_raw 0.026, mag_a_jrem 0.018, cm_atens_jrem 0.069

### Leak test (prereg §4.4.3; J-Lens set score on the steered L42 vectors, two-sided vs K = 100 J-complement contrasts)

Content set: 10 digit tokens (single-token 10–99: none); label set: [' number', 'Number', 'number', ' Number', ' digit', ' operand', 'Operand', 'digit', 'Digit', 'operand', ' Operand', ' Digit'].

| direction | set | AUROC(+ vs −) [band] | mean shift [band] | fires (95%) |
|---|---|---|---|---|
| mag_a_raw | content | 0.482 [0.491, 0.506] z -4.3 | -0.027 [-0.013, 0.008] z -4.6 | **YES** |
| mag_a_jrem | content | 0.501 [0.491, 0.506] z +0.2 | +0.001 [-0.013, 0.008] z +0.2 | no |
| cm_atens_jrem | content | 0.514 [0.469, 0.523] z +1.0 | +0.021 [-0.045, 0.032] z +1.1 | no |
| mag_a_raw | label | 0.466 [0.491, 0.507] z -8.3 | -0.039 [-0.009, 0.007] z -9.1 | **YES** |
| mag_a_jrem | label | 0.498 [0.491, 0.507] z -0.3 | -0.002 [-0.009, 0.007] z -0.4 | no |
| cm_atens_jrem | label | 0.494 [0.469, 0.523] z -0.3 | -0.007 [-0.035, 0.026] z -0.3 | no |

### G4, model side (few-shot bank, centred edit at "=", L42; nulls: K = 20 J-complement + 20 isotropic at the mag_a_jrem dose, 10 + 10 at the cm_atens_jrem dose)

| direction | greedy flip rate [jc band] [iso band] | signed value shift (expected sign) [jc] [iso] | frac moved as expected | digit log-odds drop [jc] [iso] | KL (digit pos.) |
|---|---|---|---|---|---|
| mag_a_raw | 0.000 [0.000, 0.000] z +nan [0.000, 0.000] z +nan | +0.00 [0.000, 0.000] z +nan [0.000, 0.000] z +nan | 0.00 (n 600) | +0.002 [-0.010, 0.011] z +0.1 [-0.010, 0.013] z +0.2 | 0.000 |
| mag_a_jrem | 0.000 [0.000, 0.000] z +nan [0.000, 0.000] z +nan | +0.00 [0.000, 0.000] z +nan [0.000, 0.000] z +nan | 0.00 (n 600) | -0.000 [-0.010, 0.011] z -0.3 [-0.010, 0.013] z -0.1 | 0.000 |
| cm_atens_jrem | 0.000 [0.000, 0.000] z +nan [0.000, 0.000] z +nan | +0.00 [0.000, 0.000] z +nan [0.000, 0.000] z +nan | 0.00 (n 531) | +0.016 [-0.011, 0.009] z +2.5 [-0.014, 0.010] z +2.4 | 0.000 |

Null KL (digit position): jc 0.000, iso 0.000. Retained fraction (J-removed / raw, a-magnitude): value shift 0.00, flip 0.00, digit log-odds drop -0.03. Valid digit rows: {'none': 600, 'mag_a_jrem+': 600, 'cm_atens_jrem+': 600}.

### J++ second basis (`koayon/jpp-lenses`, layer 42; `results/stage2a/jpp.json`, full table in `analysis.md`)

The J++ layer-42 map is a [5120, 5120] Jacobian for Qwen/Qwen3.6-27B (`lens_type = jacobian`, trained lens, 64 prompts).
Its top-r basis overlaps ours (n1000) at **0.68 (r\*) / 0.71 (4r\*)**, so it is a genuinely different
basis; its isotropic random band at r\* is [0.88, 1.11]. Enrichment median, n1000 → J++:

| bank | direction | r\* | 4r\* |
|---|---|---|---|
| bare (stage 1) | a_tens class-mean | 1.16 → 1.18 | 1.14 → 1.14 |
| bare (stage 1) | a_units class-mean | 1.12 → 1.13 | 1.15 → 1.10 |
| bare (stage 1) | b_tens class-mean | 1.14 → 1.22 | 1.16 → 1.15 |
| bare (stage 1) | b_units class-mean | 1.26 → 1.45 | 1.49 → 1.53 |
| bare (stage 1) | a_tens probe-weight | 0.71 → 0.72 | 0.79 → 0.82 |
| bare (stage 1) | b_units probe-weight | 0.72 → 0.80 | 0.85 → 0.91 |
| bare (stage 1) | mag_a | 0.66 → 0.82 | 0.82 → 0.88 |
| bare (stage 1) | mag_b | 0.69 → 0.79 | 0.81 → 0.89 |
| bare (stage 1) | mag_answer | 0.74 → 0.88 | 0.86 → 0.93 |
| few-shot | a_tens class-mean | 1.03 → 1.04 | 1.68 → 1.24 |
| few-shot | a_units class-mean | 0.98 → 0.94 | 1.61 → 1.20 |
| few-shot | b_tens class-mean | 1.06 → 1.12 | 1.18 → 1.14 |
| few-shot | b_units class-mean | 1.34 → 1.61 | 1.77 → 1.79 |
| few-shot | ans_tens class-mean | 1.62 → 1.53 | 1.38 → 1.30 |
| few-shot | a_tens probe-weight | 0.59 → 0.68 | 0.84 → 0.85 |
| few-shot | b_units probe-weight | 0.67 → 0.74 | 0.83 → 0.90 |
| few-shot | mag_a | 0.66 → 0.69 | 0.79 → 0.85 |
| few-shot | mag_answer | 0.77 → 0.81 | 0.83 → 0.87 |
| few-shot G4 fold 0 | mag_a_raw | 0.72 → 0.77 | 0.79 → 0.86 |
| few-shot G4 fold 0 | mag_a_jrem | 0.00 → 0.15 | 0.00 → 0.40 |
| few-shot G4 fold 0 | cm_atens_raw | 0.90 → 1.00 | 1.69 → 1.23 |
| few-shot G4 fold 0 | cm_atens_jrem | 0.00 → 0.16 | 0.00 → 0.48 |

Under J++ the information-carrying directions stay below random (probe weights 0.63–0.80 at r\*, magnitude 0.69–0.88),
and the n1000-J-removed steering directions, 0 by construction under n1000, read 0.15–0.18 at r\* and 0.39–0.48 at 4r\*
under J++: still far below the random band, so the J-removed doses are J-depleted under both lenses. The one place the
lenses disagree is the few-shot class-mean bump at 4r\* (a_tens 1.68, a_units 1.61 under n1000 → 1.24, 1.20 under J++);
only b_units (the most recent digit) is enriched under both (1.77 → 1.79).

## Reading (stage 2a; the three facts asked for, then what they mean)
1. **Leak-clean: yes.** Both J-removed directions pass the §4.4.3 test on both token sets (|z| ≤ 1.1 for `mag_a_jrem`
   and `cm_atens_jrem`, no band fires), while the raw a-magnitude direction fires as expected (content z −4.3/−4.6,
   label z −8.3/−9.1). The J-removed q90 doses are certified non-J.
2. **Causally active at the q90 dose: no, and the test is underpowered by construction.** Zero greedy flips in
   600 × 86 steered rows, value shift 0, KL at the digit position ≈ 1e-5; only `cm_atens_jrem` shows a sub-threshold
   first-digit log-odds change (z +2.5 / +2.4). The edit did fire (≈ 50–75 % of rows show bf16-visible logit changes,
   `g4_raw.pt`), but q90 − median on these directions is only **0.018–0.069 ‖h‖** (col's q90 was 0.11; the probe-type
   directions live in low-variance dimensions, so their natural q90 is tiny). His 28 % answer flips came from swapping a
   whole 18-direction class-mean subspace on one-digit operands, a far larger edit. A fair G4 needs the r11 matched
   relative-norm dose (prereg S2) or a subspace swap; **not run here**.
3. **J-depleted: under the n1000 basis, yes for the information-carrying directions; the few-shot context moves the
   class-means toward J at 4r\*.** Probe-weight (0.59–0.67 at r\*) and magnitude (0.66–0.77) directions sit below the
   random band at r\*, as in stage 1. But in the few-shot context the class-mean medians reach 1.61–1.77 at 4r\* for
   a_tens, a_units, b_units (shuffled-label reference 1.24–1.26), so by the letter of §4.2 those slots flip to IN-J at
   4r\* while staying ≈ 1.0 at r\*; the J-Lens readout rises to 0.54–0.60 with digits in the top-10 for 2–3 % of items.
   J++: the J-removed directions stay J-depleted under the J++ basis too (0.15–0.18 at r*, 0.39–0.48 at 4r*; random = 1.0), the probe-weight and magnitude directions stay below random, and the few-shot 4r* class-mean enrichment of a_tens/a_units does not survive the change of lens (1.68/1.61 → 1.24/1.20); b_units does (1.77 → 1.79).
- **The planned in-J reference failed:** even though the model answers 100 % correctly, the answer digits are only weakly
  decodable at L42 at "=" (exact 0.14) and at the space (0.10; ans_units fails G1). The answer is computed after
  layer 42 or at the digit position itself; so no "answer = IN-J" anchor exists at this layer.
- **Few-shot context lowers operand decodability** (exact a 0.415 vs 0.825 bare; b 0.612 vs 0.948): with exemplars in
  context, the "=" position carries less operand identity. His 83 %/93 % is a bare-prompt figure.
- **What this does to S10:** the operand's class-mean directions are geometrically J-marginal (depleted at r\*, enriched at
  4r\* in the few-shot context) and lens-weak, the J-removed directions are leak-clean, and causal use is untested at a
  meaningful dose. S10 cannot be filled on this evidence; the next model-side step is G4 at r11 (one extra A100 run of
  ≈ 45 min), after which the reader test (stage 2b) is or is not worth running.


---

## Stage 2a-G4 — one-digit bank, r11 / r22 doses, class-mean subspace swap (PILOT, EXPLORATORY, model side only)

**Why:** stage 2a's G4 was uninformative by dose (q90 on these directions is 0.02–0.07 ‖h‖). This run uses the
prereg's matched relative-norm doses (r11 = 0.11 ‖h_i‖, r22 = 0.22 ‖h_i‖) and Kenney's construction (a class-mean
subspace swap) on a one-digit bank where a patch at "=" can in principle change the whole answer.

**Design (`o10g4_modal_rfm.py`, `o10g4_lib.py`, tests `test_o10g4_lib.py`; control `o10g4b_modal_rfm.py`).**
- Bank: all a, b ∈ 1..9 with "+" (81) and "−" with a > b (36): **117 items**, each with 3 deterministic few-shot
  exemplars (two-digit allowed, operands disjoint from the item's), e.g. `'68 + 85 = 153\n97 - 23 = 74\n78 - 34 = 44\n1 + 1 ='`. The model answers
  1.000 correctly (greedy); first generated token is " " in 1.000. Capture at the final "=" (L42).
- Directions from the 600-item few-shot bank (`stage2a/dirs_fewshot.pt`): a_units one-vs-rest class-means (digits
  1–9) and the a-magnitude ridge direction, each raw and J-removed (top-1536 n1000 J-space). Per item, `cm_own` is the
  class-mean of the item's own digit (+Δ toward it, −Δ away); `mag` is signed by magnitude. Centred ±Δ_i.
- Swap: 9-dim span of the a_units class-means (orthonormalised), target profile = this bank's mean profile for
  a′ = a + 3 wrapped in 1..9; raw, and J-removed (class-means projected out of J first, then swapped in their span).
  Displacement / ‖h‖: raw 0.160, J-removed 0.120.
- Bands: K = 20 J-complement + 20 isotropic directions at the same per-item norm (per dose); for the swap, 20 + 20
  random 9-dim subspaces (J-complement / isotropic) at the same displacement norm. 219 conditions per item,
  16 rows per forward, one prefill + 3 cached decode steps.
- Measures: greedy flip rate vs the unsteered answer; first-digit log-odds (correct vs best other) as a signed
  ± effect and as a drop vs unsteered; signed numeric shift (mag: + = larger answer; swap: follow-exact = a′ ± b,
  moved-toward, normalised shift); KL at the digit position and at the "=" next-token position.
- Leak (prereg §4.4.3) for every condition: digit content set and number/digit/operand label set, K = 100
  J-complement contrasts at the same norm (random 9-dim J-complement subspaces for the swap). A firing dose is void.

**Run.** `rfm` / `operand_stage2a_g4`, call `fc-01M4QSZ3RTFX6DRNF62X35CNTC`, A100-80GB ×1: 938 s function time
(G4 loop 749 s) ≈ 18 min billed, ≈ $1. Control `operand_stage2a_g4b` (donor patch, below) ≈ 5 min.
Outputs: `rfm-results/10_operand_gate/stage2a_g4/{acts_onedigit_hs43.pt, g4_raw.pt, stage2a_g4.json, g4b_raw.pt,
stage2a_g4b.json}`; local `results/stage2a_g4/`.

### Results

Unsteered first-digit margin (correct digit vs best other): mean 6.65 nats, min 4.50. Enrichment of the directions
under n1000: class-means raw r\* 0.70–1.12 / 4r\* 1.04–1.79; magnitude raw r\* 0.66; J-removed 0 by construction.

### Directions (centred ±Δ_i at r11 / r22; z vs K = 20 J-complement and 20 isotropic directions at the same per-item norm)

| condition | flip + / − | flip mean [z] | first-digit log-odds signed (+ − −)/2 [z] | log-odds drop [z] | value signed effect [z] | KL digit pos. (null jc/iso) | leak |
|---|---|---|---|---|---|---|---|
| cm_own_jrem@r11 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | +0.00 [jc +0.1 / iso -0.1] | -0.04 [jc -5.4 / iso -8.0] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | clean (content z -0.1/-0.1, label z -0.0/+0.0) |
| cm_own_raw@r11 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | -0.01 [jc -0.3 / iso -0.5] | -0.03 [jc -3.9 / iso -6.0] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | clean (content z +0.5/+0.6, label z -1.0/-1.7) |
| mag_jrem@r11 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | -0.05 [jc -1.2 / iso -1.4] | -0.02 [jc -2.5 / iso -3.9] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | clean (content z -0.5/-0.5, label z +0.5/+0.5) |
| mag_raw@r11 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | -0.05 [jc -1.3 / iso -1.5] | -0.03 [jc -3.7 / iso -5.6] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | **VOID** (content z +0.2/+0.2, label z -8.5/-7.5) |
| cm_own_jrem@r22 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | +0.01 [jc +0.1 / iso -0.1] | -0.07 [jc -2.5 / iso -3.7] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | clean (content z -0.1/-0.1, label z +0.0/+0.1) |
| cm_own_raw@r22 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | -0.01 [jc -0.1 / iso -0.3] | -0.05 [jc -1.5 / iso -2.5] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | clean (content z +0.6/+0.6, label z -1.0/-1.0) |
| mag_jrem@r22 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | -0.11 [jc -1.4 / iso -1.6] | -0.09 [jc -3.4 / iso -4.9] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | clean (content z -0.5/-0.4, label z +0.4/+0.5) |
| mag_raw@r22 | 0.00 / 0.00 | 0.00 [jc nan / iso nan] | -0.09 [jc -1.2 / iso -1.3] | -0.10 [jc -4.0 / iso -5.7] | +0.00 [jc nan / iso nan] | 0.000 (0.000/0.000) | **VOID** (content z +0.2/+0.2, label z -8.6/-7.8) |

### Class-mean subspace swap (9-dim a_units span; target a' = a + 3 wrapped; z vs 20 + 20 random 9-dim subspaces at the same displacement)

| swap | follow exact (= a' ± b) [z] | moved toward [z] | normalised shift mean [z] | flip rate [z] | log-odds drop | KL digit (null jc/iso) | leak |
|---|---|---|---|---|---|---|---|
| swap_raw | 0.00 [jc nan / iso nan] | 0.00 [jc nan / iso nan] | +0.00 [jc nan / iso nan] | 0.00 [jc nan / iso nan] | +0.03 | 0.000 (0.000/0.000) | **VOID** (content z +10.5/-13.1, label z +19.9/+34.5) |
| swap_jrem | 0.00 [jc nan / iso nan] | 0.00 [jc nan / iso nan] | +0.00 [jc nan / iso nan] | 0.00 [jc nan / iso nan] | +0.01 | 0.000 (0.000/0.000) | **VOID** (content z -1.7/-3.2, label z +0.2/-0.2) |

Raw rows (`g4_raw.pt`): at r22 the mean first-digit log-odds change is −0.06 to −0.20 nats (max |Δ| ≈ 1.4), KL at the
digit position 1e-4 to 3e-4 and at the "=" position ≤ 3e-4, for the real directions and the random ones alike.

### Positive control: full donor-activation patch at "=" (`o10g4b_modal_rfm.py`, call `fc-01M4QV3M672CJGRA2P67ZTBT4Q`)

Replace the item's whole L42 vector at "=" with the activation of the item whose a = a′ (same b; same op in
0.85 of items, otherwise the "+" item with the same b). Displacement / ‖h‖: full 0.42, J-complement part
0.34, J-only part 0.25. Prediction = a′ op b.

| patch | greedy flip rate | answer = a′ op b | answer = donor's answer | moved toward | first-digit log-odds drop | KL digit pos. |
|---|---|---|---|---|---|---|
| patch_full | 0.009 | 0.00 | 0.026 | 0.00 | -0.34 | 0.003 |
| patch_jcomp | 0.000 | 0.00 | 0.026 | 0.00 | -0.07 | 0.000 |
| patch_jonly | 0.000 | 0.00 | 0.026 | 0.00 | -0.04 | 0.000 |

(The 0.026 "answer = donor's answer" rate is the base rate of items whose donor happens to share the answer; it is identical
across patches, i.e. no patch produced it.)

### Reading
1. **Leak-clean doses:** cm_own_jrem@r11, cm_own_raw@r11, mag_jrem@r11, cm_own_jrem@r22, cm_own_raw@r22, mag_jrem@r22. Void: `mag_raw` at both doses (label set z −8.5 / −7.5 and −8.6 / −7.8),
   `swap_raw` (content z +10.5 / −13.1, label z +19.9 / +34.5) and, marginally, `swap_jrem` (content mean-shift
   z −1.7, AUROC z −3.2: fires on AUROC).
2. **No leak-clean J-removed condition moves the answer at ≥ +3 z.** Flip rates are 0.00 everywhere (117 items ×
   218 steered rows: not one greedy answer changed), value effects 0, follow-exact 0. The only statistically visible
   effect is a first-digit log-odds *drop* for `cm_own_jrem` (r11: −0.04 nats, z −5.4 / −8.0; r22: −0.07, z −2.5 / −3.7)
   and `mag_jrem` (r22: −0.09, z −3.4 / −4.9) that exceeds the random bands but is ≈ 1 % of the 6.7-nat margin.
   The edits are specific (larger than random directions at the same norm) and inert (nothing in behaviour moves).
3. **Kenney's 28 % does not reproduce in this construction.** A class-mean subspace swap at "=" with a displacement
   of 0.16 ‖h‖ (raw) changes no answer here; the raw swap also leaks heavily into the J-Lens, so even a positive
   would not have been a non-J result. The full donor patch, a 0.42 ‖h‖ replacement of the entire vector, flips 1 of 117 answers, never yields a′ op b, and lowers the first-digit margin by 0.34 of 6.65 nats; its J-complement and J-only parts flip none. **At layer 42, the "=" position does not carry the answer-determining operand information in this few-shot format**: the model reads the operands from their own positions in later layers. So no edit at this site, J-removed or not, can show causal use, and the zeros above are a property of the test site rather than of the directions. A causal test of operand identity needs edits at the operand token positions (or a multi-layer patch), which is outside this stage's scope; Kenney's 28 % was presumably obtained with a different site or layer set. **Verdict for S10: no leak-clean J-removed condition moves the answer at ≥ +3 z, and the site cannot produce a positive, so operand identity at "=" stays decodable-but-unused at L42 and does not fill the S10 slot.**
