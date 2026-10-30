"""bk_exp03_smoke: on the J-removed direction only the NLA still follows the feature (exp 03 toy)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import style  # noqa: E402
from style import C, ROOT  # noqa: E402

M = ROOT / "experiments" / "03_pipeline_smoke" / "results" / "metrics.json"
READERS = [("NLA@42", "NLA", "nla"), ("OL@42", "Oracle Lens", "ol"), ("JLens@42", "J-lens", "jl"), ("probe@42", "probe", "probe")]
REC = {("NLA@42", "raw"): 0.91, ("OL@42", "raw"): 0.84, ("JLens@42", "raw"): 0.84,
       ("NLA@42", "jrem"): 0.60, ("OL@42", "jrem"): 0.45, ("JLens@42", "jrem"): 0.46}


def main() -> None:
    ff = json.loads(M.read_text())["follow_feature"]
    checks = []
    fig, ax = plt.subplots(figsize=style.FULL)
    w = 0.36
    for i, (key, lab, c) in enumerate(READERS):
        for j, d in enumerate(("raw", "jrem")):
            q = ff[f"{key}|{d}"]["q99"]
            p, lo, hi = q["auroc"], q["lo"], q["hi"]
            xi = i + (j - 0.5) * w
            face = C[c] if d == "raw" else "white"
            ax.bar(xi, p, w * 0.92, color=C.get(f"{c}_pale", C["line"]) if d == "raw" else C[c], zorder=2,
                   edgecolor=C[c], lw=0)
            ax.errorbar(xi, p, yerr=[[p - lo], [hi - p]], color=C["ink"], capsize=3, lw=1.1, zorder=3)
            ax.text(xi, hi + 0.012, f"{p:.2f}", ha="center", va="bottom", fontsize=12.5,
                    fontweight="bold" if d == "jrem" else None)
            if d == "jrem":
                ax.text(xi, 0.04, f"{q['paired_win']:.0%}\nmoved", ha="center", va="bottom", fontsize=11.5,
                        color="white")
            if (key, d) in REC:
                checks.append({"quantity": f"FollowFeature q99 {key} {d}", "computed": round(p, 3), "record": REC[(key, d)],
                               "where": "exp 03 RESULTS FollowFeature table", "match": round(p, 2) == REC[(key, d)]})
    ax.axhline(0.5, color=C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(-0.55, 0.505, "chance", va="bottom", fontsize=11.5, color=C["muted"])
    ax.set_xticks(range(len(READERS)), [r[1] for r in READERS])
    ax.set_xlim(-0.6, 3.85)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("follows the pushed feature\n(AUROC, 0.5 = chance)")
    ax.legend(handles=[Patch(color=C["line"], label="push along the full French-vs-German direction (pale)"),
                       Patch(color=C["muted"], label="push with the J-space removed (dark)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=12.5)
    style.save(
        fig, "bk_exp03_smoke",
        title="With the J-space removed from the push, only the NLA still follows the feature (toy, n=40 pairs)",
        caption=("Pipeline smoke test on a toy in-J feature (French vs German, 40 content-matched pairs, 80 items): the layer-42 "
                 "activation is pushed along the class-mean direction at the q99 dose, either full or with the top-384 J-space "
                 "removed, and each reader's two-candidate log-odds is scored by AUROC (mean of 3 templates; 95% CI, 2,000 "
                 "pair-clustered bootstraps). Labels in the dark bars: share of items (paired win) whose log-odds moved toward the pushed "
                 "class. Oracle Lens and the J-lens move against the push. Caveat: plumbing check, not a result; the feature is "
                 "trivially present in the text."),
        section="backup:exp03", role="backup", sources=["experiments/03_pipeline_smoke/results/metrics.json"], checks=checks,
        notes="Directions are 2-fold cross-fitted by pair; null = equal-norm random direction (AUROC 0.47 to 0.54).",
    )


if __name__ == "__main__":
    main()
