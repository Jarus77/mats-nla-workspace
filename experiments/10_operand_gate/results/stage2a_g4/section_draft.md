

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

{G4B_BLOCK}

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
   would not have been a non-J result. {G4B_FACT}
