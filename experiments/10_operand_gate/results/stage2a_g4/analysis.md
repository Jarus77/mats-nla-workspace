One-digit bank: n = 117, model correct unsteered 1.000, first token is the space 1.000; median ‖h‖ 89.8; swap displacement / ‖h‖: raw 0.160, J-removed 0.120; 219 conditions per item.

Direction enrichment (n1000): class-means raw r* 0.70–1.12, raw 4r* 1.04–1.79, J-removed 4r* 0.00; magnitude raw r* 0.66, J-removed 4r* 0.00.

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

Wall time: G4 748.9 s, total 937.7 s.
