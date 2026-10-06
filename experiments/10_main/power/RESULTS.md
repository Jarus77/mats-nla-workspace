# TOST power simulation for the pre-registered negatives N1–N5 (D27.4)

**Bottom line.** I defined the equivalence test on the statistic the null bands are built from: the per-item paired effect e_i, compared with the K = 100 band on the same fresh items. On that statistic the test is easily reachable. If the true effect is 0, IUT power is 1.00 at every n from 120 to 960; at 0.5×SESOI it is 0.97 at n = 120 and 1.00 from 240 up. IUT here means all five OL descriptive families pass, each against both bands. **n stays 240 col passages, the primaries' n**, so D27's "same n" holds and no increase is needed. The only case where more items help is if N5 really anti-tracks, as its pilot point estimate suggests. In that case TOST *should* refuse to call it "flat", and non-superiority (the substantive claim) still has power 1.00.

All inputs come from the pilot split, which is excluded from the main analysis. CPU only. Seed 20261006, 2,000 simulated experiments per cell.

```
../../../.venv/bin/python tost_sim.py                    # main: OL N1–N5  -> tost_sim.json, tost_sim_stdout.txt
../../../.venv/bin/python tost_sim.py --noscale          # sensitivity: nulls left at the larger pilot norm
../../../.venv/bin/python tost_sim.py --dirinflate 1.3   # sensitivity: direction-level SD x 1.3
../../../.venv/bin/python tost_sim.py --reader nla       # S9: declared NLA negatives
```
Re-running reproduces every number below exactly (checked: two runs gave identical output).

## 1. The test (written into prereg §3.3)

- **Statistic.** It is the one the primaries' null band uses (prereg §3.1). Per item, e_i = (LO_i(+Δ) − LO_i(−Δ))/2, where the family LO is the mean over the family's templates. The test statistic is
  T_f = ē_f(real) − (1/K) Σ_k ē_f(null_k),
  with K = 100 per band, on the same fresh items and at the same n as the primaries.
- **SE.** SE² = Var_i(e_real,i − ē_band,i)/n + Var_k(ē_k)/K. The first term is item-level sampling of the paired difference. The second is direction-level sampling of the band centre.
- **TOST.** A family passes iff T − 1.645·SE > −δ_f and T + 1.645·SE < δ_f, against **both** bands (isotropic and J-complement). "Flat" is claimed only if all five families pass (intersection-union test, IUT). Non-superiority means only the upper side holds.
- **Why not AUROC.** The rev-4 test was TOST on cross-item AUROC with an item bootstrap. Its SE comes from items only. The unpaired Hanley–McNeil SE (≈ 0.037 at n = 240) cannot close ±0.05. The paired item bootstrap is much tighter (0.004–0.005 at n = 240, from this pilot), but it still ignores how much random directions move these templates. The ē test includes that band variability and is on the same scale as the primaries.

## 2. SESOI mapped onto ē

AUROC 0.55 corresponds to a fraction of probe of 0.10. δ_f is the ē shift that, applied as a pure location shift (+δ to LO(+Δ), −δ to LO(−Δ)), moves the cross-item AUROC of the pilot's LO distribution at the P1/P2 cell from 0.50 to 0.55. It is computed empirically by bisection over distinct-item pairs. Normal theory gives δ = Φ⁻¹(0.55)·SD(LO)/√2 = 0.089·SD(LO), within 15% of the empirical value.

| OL family | δ_f (nats, frozen) | normal-theory δ | SD(LO) | direction-level SD iso / Jc | **δ / direction-level SD** iso / Jc | pilot ē(real) − Jc band centre |
|---|---|---|---|---|---|---|
| N1 count_qual | 0.259 | 0.297 | 3.34 | 0.114 / 0.081 | **2.3 / 3.2** | −0.056 (−0.22δ) |
| N2 count_num | 0.124 | 0.131 | 1.47 | 0.026 / 0.024 | **4.8 / 5.1** | +0.011 |
| N3 decision | 0.214 | 0.240 | 2.70 | 0.110 / 0.055 | **1.9 / 3.9** | +0.003 |
| N4 f07_count | 0.119 | 0.126 | 1.42 | 0.038 / 0.037 | **3.1 / 3.2** | −0.041 (−0.34δ) |
| N5 f07_break | 0.233 | 0.253 | 2.84 | 0.103 / 0.091 | **2.3 / 2.6** | −0.139 (−0.60δ; −0.82δ vs iso) |

The direction-level SD is the part of the band SD that does not shrink with n. It is estimated as SD_between minus the within-direction item SE, from the pilot's K = 20 null blocks, rescaled to the q90 norm. **The SESOI is 2–5 band SDs**, so a random direction does not by itself breach the bound, and the bound is reachable in principle.

S9 (NLA negatives, same mapping): δ = 0.170 (count_qual), 0.165 (count_num), 0.087 (decision) and 0.058 (f07_break) nats, which is 1.8–5.3 band SDs.

## 3. Simulation model

- **Items.** Each simulated experiment bootstraps n passages from the 60 pilot passages. The draw is joint across families and bands, so cross-family and real-vs-null item correlations are kept.
- **Null band.** Each band gets K = 100 fresh directions:
  `null_k,i = μ_band + a_k + R[j, item_i]`.
  - a_k ~ MVN(0, Σ_dir) is the direction-level effect. Σ_dir is the pilot's between-direction covariance minus its item-noise part, projected to be positive semi-definite. It is fixed in n.
  - R is the direction-centred pilot residual, which keeps the item component shared across directions. A pilot column j is drawn per (k, i).
- **Real direction.** `real_i = μ_Jc + θ + ρ[item_i]`, where ρ is the centred per-item residual of the pilot's real cell.
- **Scenarios.**
  - θ = 0.
  - θ = 0.5·δ_f.
  - θ = the pilot point estimate.
  - θ = δ_f, as a size check.
  - "B0": θ = 0, but the real direction carries its own a_r ~ MVN(0, Σ_dir,Jc). This is the case where it behaves exactly like a random J-complement direction, and its idiosyncrasy counts as effect, not noise.
- **Dose matching.** The pilot nulls at this cell exist only at the reg_raw q90 norm (9.465), not at the real cell's 7.098. They are rescaled by 7.098/9.465 = 0.75, assuming a linear dose response: OL col is linear in dose (R5), and the band SD at r11 ≈ 9.9 is within about 5% of the band SD at 9.465. `--noscale` drops the rescaling.

## 4. Results: OL N1–N5. P(all five families pass TOST against both bands)

| True effect (each family) | n = 120 | **240** | 480 | 960 |
|---|---|---|---|---|
| θ = 0 | 1.00 | **1.00** | 1.00 | 1.00 |
| θ = 0.5·SESOI | 0.97 | **1.00** | 1.00 | 1.00 |
| θ = pilot point estimate (N5 anti-tracks) | 0.65 | **0.82** | 0.92 | 0.97 |
| B0: real direction exchangeable with a random Jc direction | 0.89 | **0.91** | 0.93 | 0.93 |
| IUT non-superiority, worst row above | 0.96 | **0.97** | 0.98 | 0.98 |

- **Size.** At θ = δ_f the empirical per-family rejection rate against the Jc band is 0.04–0.06, so the test is correctly sized and the SE formula is not anti-conservative.
- **Per family.** At n = 240, every family has power ≥ 0.99 under θ = 0 and θ = 0.5·SESOI. In the pilot-estimate row, N5 alone fails, and only on the iso band's lower bound (0.82). In B0 the weakest family is N5 (0.94).

**Sensitivities, at n = 240 (n = 120 in brackets):**

| Variant | θ = 0 | θ = 0.5·SESOI | pilot estimate | B0 |
|---|---|---|---|---|
| main (nulls rescaled to q90) | 1.00 (1.00) | 1.00 (0.97) | 0.82 (0.65) | 0.91 (0.89) |
| `--noscale` (nulls at the 33% larger norm; conservative) | 1.00 (1.00) | 0.99 (0.91) | 0.47 (0.37) | 0.67 (0.64) |
| `--dirinflate 1.3` (direction SD × 1.3, roughly the K = 20 estimation error) | 1.00 (1.00) | 1.00 (0.96) | 0.72 (0.58) | 0.74 (0.71) |

**S9, NLA negatives (`--reader nla`):** 1.00 at θ = 0. At θ = 0.5·SESOI: 0.98 at n = 120 and 1.00 from 240 up. Pilot estimate: 0.89 at n = 120 and 0.99 at n = 240. B0: 0.60 / 0.69 / 0.73 / 0.73. NLA f07_break's δ is only about 2 band SDs.

## 5. Reading

1. **n needed for ≥ 80% power:** n = 120 already suffices at θ = 0 and at θ = 0.5·SESOI, and this holds under every sensitivity. 240 gives ≥ 0.99. **Chosen n = 240**, the primaries' n, as D27 requires ("same n"). There is no need to go to 480.
2. **What n cannot fix.** In row B0 power plateaus (0.91–0.93 in the main run; 0.67–0.74 under the sensitivities) because the direction-level SD does not shrink with n. If the real direction's idiosyncrasy is counted as effect, the test can fail by bad luck of the direction itself. That is the honest meaning of a SESOI that is only 2–5 band SDs. Raising n does not help; only K for the band centre or a wider bound would. We keep the bound (D27: "raise n rather than loosen the bound") and K = 100.
3. **N5.** The pilot point estimate (ē −0.16 against band centres of −0.02 (Jc) and +0.03 (iso)) sits 0.6–0.8 δ below the band. If that is real, TOST's lower bound fails at a rate that n only slowly removes: 0.82 → 0.97 from 240 to 960, and lower under the sensitivities. This is the outcome prereg §3.3 already predicts and words as "anti-tracking". Non-superiority ("does not express +col") is unaffected (power 1.00).
4. **Caveats.**
   - The direction-level covariance comes from 20 pilot directions (relative SE ≈ 16%), at a norm 33% larger and rescaled. Both are covered by the sensitivities.
   - Direction effects are modelled as normal.
   - Items are a bootstrap of 60 pilot passages. The fresh items use the same generator.
   - δ_f is frozen from the pilot LO spread. If the fresh LO spread differs, the AUROC equivalent of δ_f is reported but δ is not re-estimated.

## Files
- `tost_sim.py`: the simulation, seeded and CPU only. On first run it extracts the pilot arrays from `08_col_pilot/results/08_col_pilot/r5_{ol,nla}_chunk*.pt` and caches them in `pilot_inputs*.npz`.
- `tost_sim.json`, `tost_sim_noscale.json`, `tost_sim_dir1.3.json`, `tost_sim_nla.json`: full outputs (δ, band decomposition, per-band and per-family power, IUT).
- `*_stdout.txt`: the printed tables.
