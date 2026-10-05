"""Local summary of all results (run after `modal volume get mats-results 02_nonj_features/<dir>`).

    python summarize.py <local_dir_with_probe_results_causal_leak_lensbag_selfreport>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FEATS = ["col", "depth", "pred_pos", "tense", "lang", "col_report", "topic", "nexttok", "next_pos"]


def load(p: Path) -> dict | None:
    return json.load(open(p)) if p.exists() else None


def main(root: str) -> None:
    R = Path(root)
    rstar = load(R / "rstar.json")
    rs = rstar["layers"]["42"]["r_star"] if rstar else None
    print("r* (J-lens L42) =", rs)
    for f in FEATS:
        pr = load(R / "probe_results" / f"{f}.json")
        if not pr:
            continue
        k = "acc" if pr["kind"] == "cls" else "r2"
        print(f"\n===== {f}  n={pr['n_rows']} docs={pr['n_docs']} chance={pr['chance']}")
        print("  (i) by hs:", {h: f"{v['raw'][k]:.3f}±{v['raw'][k + '_sd']:.3f} ci{[round(x, 3) for x in v['raw'].get(k + '_ci95', [])]} ctrl {v['control_task'][k]:.3f}"
                              for h, v in pr["layers"].items() if h in ("40", "42", "43", "44")})
        for h in ("43",):
            rem = pr["removal"][h]
            base = pr["layers"][h]["raw"][k]
            grid = sorted(int(x) for x in rem["R2v_spectrum_energy_frac"])
            print("  (ii) retrain  :", {r: round(rem[f"R2v_r{r}_retrain"][k], 3) for r in grid})
            print("       rand     :", {r: round(rem[f"R2v_rand{r}_retrain"][k], 3) for r in grid})
            print("       frozen   :", {r: round(rem[f"R2v_r{r}_frozen"][k], 3) for r in grid})
            print("       randfroz :", {r: round(rem[f"R2v_rand{r}_frozen"][k], 3) for r in grid})
            print("       R1 k=25  :", round(rem["R1_retrain"][k], 3), round(rem["R1_frozen"][k], 3), "base", round(base, 3))
            en = rem["enrichment"]
            print("       enrich steer:", {r: round(en[str(r)]["steer"]["M_enrichment"][0], 2) for r in grid},
                  "acts:", {r: round(en[str(r)]["activations_centered"]["M_enrichment"], 2) for r in grid})
            r3 = rem["R3"]
            print("       R3:", {kk: round(v, 3) for kk, v in r3.items() if isinstance(v, float)})
            L = pr["lens"][h]["std"]
            print("  (iii)", {kk: round(v, 4) if isinstance(v, float) else v for kk, v in L.items()})
        lb = load(R / "lensbag" / f"{f}_hs43.json")
        if lb:
            print("  lensbag:", {n: round(lb[n][k], 3) for n in ("lens_bag", "cur_tok", "local4", "local4+lens") if n in lb})
        ca = load(R / "causal" / f"{f}_hs43.json")
        if ca:
            for dn, e in ca["effects"].items():
                print(f"  (iv) {dn}: probe {e['probe_signed_effect']:.3f} {[round(x, 3) for x in e['probe_effect_ci95']]} rand max {e['random_abs_effect_max']:.3f}",
                      {kk: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for kk, v in e.items() if kk.startswith("jrem") or kk.startswith("jorth")},
                      "KL", round(e["probe_kl_plus"], 4), "randKL", round(e["random_kl_plus_mean"], 4),
                      "steer/norm", round(ca["conditions"][f"probe+{dn}"]["steer_over_resid_norm"], 3))
        lk = load(R / "leak" / f"{f}.json")
        if lk:
            for L, s in lk["summary"].items():
                keys = [x for x in s if x.endswith("2xp99")]
                print(f"  leak L{L}: base top10 {s['baseline_pos_top10']:.3f}",
                      {x: (round(s[x]["d_pos_logrank"], 3), round(s[x]["pos_top10"], 3), round(s[x]["top10_overlap_with_unsteered"], 2)) for x in keys})
        sr = load(R / "selfreport" / f"{f}.json")
        if sr:
            print("  (v)", {kk: v for kk, v in sr.items() if kk not in ("examples",) and not kk.endswith("ci95")})


if __name__ == "__main__":
    main(sys.argv[1])
