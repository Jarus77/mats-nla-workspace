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

### J++ second basis (koayon/jpp-lenses, layer 42; enrichment median, n1000 → J++)

Skipped: None

Wall time: capture 52.8 s, G4 2542.6 s, total 2917.3 s.
