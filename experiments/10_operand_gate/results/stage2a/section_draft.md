

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

{JPP_BLOCK}

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
   J++: {JPP_FACT}
- **The planned in-J reference failed:** even though the model answers 100 % correctly, the answer digits are only weakly
  decodable at L42 at "=" (exact 0.14) and at the space (0.10; ans_units fails G1). The answer is computed after
  layer 42 or at the digit position itself; so no "answer = IN-J" anchor exists at this layer.
- **Few-shot context lowers operand decodability** (exact a 0.415 vs 0.825 bare; b 0.612 vs 0.948): with exemplars in
  context, the "=" position carries less operand identity. His 83 %/93 % is a bare-prompt figure.
- **What this does to S10:** the operand's class-mean directions are geometrically J-marginal (depleted at r\*, enriched at
  4r\* in the few-shot context) and lens-weak, the J-removed directions are leak-clean, and causal use is untested at a
  meaningful dose. S10 cannot be filled on this evidence; the next model-side step is G4 at r11 (one extra A100 run of
  ≈ 45 min), after which the reader test (stage 2b) is or is not worth running.
