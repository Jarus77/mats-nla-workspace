"""bk_exp10_tost_power: 240 passages are enough to call line position flat; direction noise sets a ceiling more items cannot lift."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

P = S.ROOT / "experiments" / "10_main" / "power"


def main():
    main_ = json.loads((P / "tost_sim.json").read_text())
    sens = {"nulls at the larger pilot norm": json.loads((P / "tost_sim_noscale.json").read_text()),
            "direction spread x 1.3": json.loads((P / "tost_sim_dir1.3.json").read_text())}
    ns = sorted(int(n) for n in main_["power"])
    nsim = main_["nsim"]
    scen = [("A0", "true effect 0", S.C["probe"]), ("Ahalf", "true effect = half the smallest effect of interest", S.C["muted"]),
            ("Apilot", "true effect = pilot estimate (one family anti-tracks)", S.C["text"]),
            ("B0", "real direction behaves like a random direction", S.C["ol"])]
    fig, ax = S.plt.subplots(figsize=S.FULL)
    x = np.log2(ns)
    vals = {}
    for key, lab, c in scen:
        ys = [main_["power"][str(n)][key]["iut_all5_both_bands"] for n in ns]
        vals[key] = ys
        ci = [S.wilson(int(round(y * nsim)), nsim) for y in ys]
        ax.fill_between(x, [c_[1] for c_ in ci], [c_[2] for c_ in ci], color=c, alpha=0.18, lw=0)
        ax.plot(x, ys, "-o", color=c, lw=2.4, ms=8, label=lab)
    for (lab, d), ls in zip(sens.items(), (":", "--")):
        ys = [d["power"][str(n)]["B0"]["iut_all5_both_bands"] for n in ns]
        vals[f"B0 {lab}"] = ys
        ax.plot(x, ys, ls, color=S.C["ol_pale"], lw=2)
        ax.text(x[-1] + 0.08, ys[-1], f"same, {lab}", color=S.C["ol"], fontsize=11.5, va="center")
    ax.axhline(0.8, color=S.C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(x[0] - 0.05, 0.805, "80% power", color=S.C["muted"], fontsize=12, va="bottom")
    ax.axvline(np.log2(240), color=S.C["line"], lw=1.2, zorder=0)
    ax.text(np.log2(240) + 0.04, 0.42, "chosen n = 240\n(the primaries' n)", color=S.C["muted"], fontsize=12, va="bottom")
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlim(x[0] - 0.15, x[-1] + 1.55)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("passages in the main run (log scale)")
    ax.set_ylabel("P(all five Oracle Lens families\npass the equivalence test)")
    ax.legend(loc="lower right", fontsize=12, bbox_to_anchor=(1.0, 0.0))
    rec = {("A0", 240): 1.00, ("Ahalf", 120): 0.97, ("Ahalf", 240): 1.00, ("Apilot", 120): 0.65, ("Apilot", 240): 0.82,
           ("Apilot", 480): 0.92, ("Apilot", 960): 0.97, ("B0", 120): 0.89, ("B0", 240): 0.91, ("B0", 480): 0.93, ("B0", 960): 0.93}
    checks = []
    for (k, n), r in rec.items():
        v = main_["power"][str(n)][k]["iut_all5_both_bands"]
        checks.append({"quantity": f"IUT power {k} n={n}", "computed": round(v, 3), "record": r, "where": "10_main/power/RESULTS.md section 4",
                       "match": abs(v - r) < 0.006})
    for lab, d, r in (("noscale", sens["nulls at the larger pilot norm"], 0.67), ("dir1.3", sens["direction spread x 1.3"], 0.74)):
        v = d["power"]["240"]["B0"]["iut_all5_both_bands"]
        checks.append({"quantity": f"B0 n=240 {lab}", "computed": round(v, 3), "record": r, "where": "10_main/power/RESULTS.md sensitivities", "match": abs(v - r) < 0.006})
    S.save(fig, "bk_exp10_tost_power",
           title="240 passages are enough to call line position flat; direction-to-direction noise caps power that more items cannot lift",
           caption=("Simulated power (2,000 experiments per point, K = 100 random directions per band, inputs bootstrapped from the "
                    "60-passage pilot) for the pre-registered equivalence test that Oracle Lens's five descriptive families are flat "
                    "against both random-direction bands. Shading: Wilson 95% Monte Carlo interval. With no true effect power is 1.00 "
                    "from n = 120; if the real direction behaves like a random one (violet) power plateaus near 0.93, and lower under "
                    "the two sensitivity runs (pale dashed), because the direction-level spread does not shrink with n. "
                    "Caveat: a simulation from pilot data, not a measurement."),
           section="backup:exp10", role="backup",
           sources=["experiments/10_main/power/tost_sim.json", "experiments/10_main/power/tost_sim_noscale.json",
                    "experiments/10_main/power/tost_sim_dir1.3.json"],
           checks=checks,
           notes=("Statistic: per-item paired effect e_i = (LO(+D) - LO(-D))/2 minus the band mean; TOST at +/- delta_f (the shift "
                  "moving AUROC 0.50 -> 0.55), intersection-union over five families and both bands. Values are the simulation's own "
                  "outputs (no raw item data involved). Series: " + json.dumps({k: [round(v, 3) for v in vs] for k, vs in vals.items()})))
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
