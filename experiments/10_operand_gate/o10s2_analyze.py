"""Stage 2a (CPU): render results/stage2a/{stage2a.json, jpp.json} as the markdown tables for RESULTS.md.

    ../../.venv/bin/python o10s2_analyze.py > results/stage2a/analysis.md
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results" / "stage2a"
o = json.loads((R / "stage2a.json").read_text())
jpp = json.loads((R / "jpp.json").read_text()) if (R / "jpp.json").exists() else None


def f2(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def band(b: dict) -> str:
    return f"[{f2(b['band_lo'], 3)}, {f2(b['band_hi'], 3)}] z {b['z']:+.1f}"


print("### Few-shot bank: G1–G3 (n = 600, 5-fold CV; enrichment r*/4r*, random = 1.0)\n")
rb = o["random_band_iso_K100"]
print(f"Random band at r*: [{f2(rb['384']['p2.5'])}, {f2(rb['384']['p97.5'])}]; at 4r*: [{f2(rb['1536']['p2.5'])}, {f2(rb['1536']['p97.5'])}].\n")
print("| slot | G1 acc | macro AUROC | G2 class-mean (median) | G2 probe-weight | shuffled-label ref | G3 lens stat [null 95%] | true digit in top-10 | label (§4.2 letter) |")
print("|---|---|---|---|---|---|---|---|---|")
for slot, s in o["slots"].items():
    g1, g2, g3 = s["G1"], s["G2"], s["G3"]
    print(f"| {slot} | {f2(g1['acc'])} | {f2(g1['macro_auroc'])} | "
          f"{f2(g2['384']['class_mean']['median'])} / {f2(g2['1536']['class_mean']['median'])} | "
          f"{f2(g2['384']['probe_weight']['median'])} / {f2(g2['1536']['probe_weight']['median'])} | "
          f"{f2(g2['384']['shuffled_label_class_mean']['median'])} / {f2(g2['1536']['shuffled_label_class_mean']['median'])} | "
          f"{f2(g3['stat'], 3)} [{f2(g3['null_band95'][0], 3)}, {f2(g3['null_band95'][1], 3)}] | {f2(g3['true_label_in_top10'])} | **{s['label']}** |")
print()
for k in ("exact_a", "exact_b", "exact_ans", "exact_ans@space"):
    print(f"- {k}: {f2(o[k]['acc'], 3)} [{f2(o[k]['ci95'][0], 3)}, {f2(o[k]['ci95'][1], 3)}]")
print(f"- few-shot model answers correctly (greedy, unsteered): {f2(o['fewshot_model_correct'], 3)}; first generated token is the space: {f2(o['fewshot_first_token_is_space'], 3)}")
for name, m in o["magnitude"].items():
    print(f"- magnitude {name}: CV r {f2(m['cv_pearson_r'])}, MAE {f2(m['cv_mae'], 1)}; enrichment {f2(m['enrichment']['384'])} / {f2(m['enrichment']['1536'])}")
print(f"- G4 directions (cross-fit folds): enrichment " + "; ".join(
    f"{k} {f2(v['384'])} / {f2(v['1536'])}" for k, v in o["g4_dirs_enrichment"]["0"].items()))
print(f"- doses (q90 − median, per fold): {o['doses']}; ‖Δ‖/median‖h‖: " + ", ".join(f"{k} {f2(v, 3)}" for k, v in o["dose_over_median_norm"].items()))

print("\n### Leak test (prereg §4.4.3; J-Lens set score on the steered L42 vectors, two-sided vs K = 100 J-complement contrasts)\n")
print(f"Content set: {o['leak_sets']['content_n']} digit tokens (single-token 10–99: {o['leak_sets']['multi_digit_single_tokens'] or 'none'}); label set: {o['leak_sets']['label']}.\n")
print("| direction | set | AUROC(+ vs −) [band] | mean shift [band] | fires (95%) |")
print("|---|---|---|---|---|")
for sname, d in o["leak"].items():
    for nm, r in d.items():
        print(f"| {nm} | {sname} | {f2(r['auroc'], 3)} {band(r['band_auroc'])} | {r['mean_shift']:+.3f} {band(r['band_mean_shift'])} | {'**YES**' if r['fires95'] else 'no'} |")

print("\n### G4, model side (few-shot bank, centred edit at \"=\", L42; nulls: K = 20 J-complement + 20 isotropic at the mag_a_jrem dose, 10 + 10 at the cm_atens_jrem dose)\n")
g = o["g4"]
print("| direction | greedy flip rate [jc band] [iso band] | signed value shift (expected sign) [jc] [iso] | frac moved as expected | digit log-odds drop [jc] [iso] | KL (digit pos.) |")
print("|---|---|---|---|---|---|")
for nm in ("mag_a_raw", "mag_a_jrem", "cm_atens_jrem"):
    fl, vs, dr = g["flip"][nm], g["value_shift"][nm], g["digit_logodds_drop"][nm]
    print(f"| {nm} | {f2(fl['flip_rate'], 3)} {band(fl['band_jc'])} {band(fl['band_iso'])} | {vs['mean_shift']:+.2f} {band(vs['band_jc'])} {band(vs['band_iso'])} | "
          f"{f2(vs['frac_moved_expected'])} (n {vs['n_valid']}) | {dr['drop']:+.3f} {band(dr['band_jc'])} {band(dr['band_iso'])} | {f2(g['kl_mean'][nm], 3)} |")
print(f"\nNull KL (digit position): jc {f2(g['kl_mean_null']['jc'], 3)}, iso {f2(g['kl_mean_null']['iso'], 3)}. "
      f"Retained fraction (J-removed / raw, a-magnitude): value shift {f2(g['retained_fraction_jrem_over_raw']['value_shift'])}, "
      f"flip {f2(g['retained_fraction_jrem_over_raw']['flip'])}, digit log-odds drop {f2(g['retained_fraction_jrem_over_raw']['digit_logodds_drop'])}. "
      f"Valid digit rows: {o['g4_valid_digit_rows']}.")

print("\n### J++ second basis (koayon/jpp-lenses, layer 42; enrichment median, n1000 → J++)\n")
if jpp is None or "skipped" in jpp:
    print(f"Skipped: {jpp}")
else:
    print(f"Basis overlap with n1000: r* {f2(jpp['overlap_with_n1000']['384'], 3)}, 4r* {f2(jpp['overlap_with_n1000']['1536'], 3)}. "
          f"Random band under J++ at r*: [{f2(jpp['random_band_jpp']['384']['p2.5'])}, {f2(jpp['random_band_jpp']['384']['p97.5'])}].\n")
    print("| bank | direction | r* n1000 → J++ | 4r* n1000 → J++ |")
    print("|---|---|---|---|")
    for bank, res in jpp["dirs"].items():
        if not isinstance(res, dict):
            print(f"| {bank} | {res} | | |")
            continue
        for k, v in res.items():
            if k == "magnitude":
                for mk, mv in v.items():
                    print(f"| {bank} | {mk} | {f2(mv['384']['n1000'])} → {f2(mv['384']['jpp'])} | {f2(mv['1536']['n1000'])} → {f2(mv['1536']['jpp'])} |")
            elif k == "g4_dirs":
                for fo, dd in v.items():
                    for dk, dv in dd.items():
                        print(f"| {bank} | g4 fold {fo} {dk} | {f2(dv['384']['n1000'])} → {f2(dv['384']['jpp'])} | {f2(dv['1536']['n1000'])} → {f2(dv['1536']['jpp'])} |")
            else:
                for kk, vv in v.items():
                    print(f"| {bank} | {k} {kk} | {f2(vv['384']['n1000'])} → {f2(vv['384']['jpp'])} | {f2(vv['1536']['n1000'])} → {f2(vv['1536']['jpp'])} |")
print(f"\nWall time: capture {o['t_capture_s']} s, G4 {o['t_g4_s']} s, total {o['t_total_s']} s.")
