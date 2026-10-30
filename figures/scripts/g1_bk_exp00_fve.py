"""bk_exp00_fve: the NLA works as published; text-only explanations reconstruct far less (exp 00)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402
from style import C, ROOT  # noqa: E402

R = ROOT / "experiments" / "00_nla_gate" / "results"
ROWS = [  # (cond key or 'textnla', label, colour, record)
    ("iter400_nothink", "NLA (RL step 400), the shipped reader", C["nla"], 0.761),
    ("iter300_nothink", "NLA (RL step 300)", C["nla_pale"], 0.759),
    ("avbase_nothink", "NLA warm start, before RL", C["nla_pale"], 0.459),
    ("textnla", "text model, writing from the source text", C["text"], 0.276),
    ("source_tail_900tok", "source text itself (last 900 tokens)", C["text_pale"], 0.208),
    ("source_tail_64tok", "source text itself (last 64 tokens)", C["text_pale"], 0.151),
    ("iter400_shuffled_act", "NLA explanation of a different activation", C["muted"], -0.903),
]


def main() -> None:
    fve = json.loads((R / "fve_gate.json").read_text())
    conds = {c["cond"]: c for c in fve["conds"]}
    tx = json.loads((R / "score_gate_textnla_lpcheck.json").read_text())["textnla"]
    off = json.loads((R / "offbyone.json").read_text())
    vals, checks = [], []
    for key, lab, col, rec in ROWS:
        if key == "textnla":
            per, mean = np.array(tx["per_example_mean"]), tx["mean"]
        else:
            per, mean = np.array(conds[key]["per_example"]), conds[key]["fve_mean_of_samples"]
        lo, hi = style.bootstrap_ci(np.mean, per)
        vals.append((lab, col, mean, lo, hi, len(per)))
        checks.append({"quantity": f"FVE {key}", "computed": round(mean, 3), "record": rec,
                       "where": "exp 00 RESULTS FVE table", "match": round(mean, 3) == rec})
    checks.append({"quantity": "FVE block-42 vs block-41 capture (off-by-one test)",
                   "computed": f"{off['hs43']['fve_self']:.3f} vs {off['hs42']['fve_self']:.3f}", "record": "0.760 vs 0.695",
                   "where": "exp 00 RESULTS K0", "match": (round(off['hs43']['fve_self'], 3), round(off['hs42']['fve_self'], 3)) == (0.760, 0.695)})

    fig, ax = plt.subplots(figsize=style.FULL)
    y = np.arange(len(vals))[::-1]
    for yi, (lab, col, m, lo, hi, n) in zip(y, vals):
        ax.barh(yi, m, 0.62, color=col, zorder=2)
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], color=C["ink"], capsize=3, lw=1.2, zorder=3)
        if m >= 0:
            ax.text(hi + 0.02, yi, f"{m:.2f}", va="center", ha="left", fontsize=13, fontweight="bold")
        else:
            ax.text(m / 2, yi, f"{m:.2f}", va="center", ha="center", fontsize=13, fontweight="bold", color="white")
    ax.set_yticks(y, [v[0] for v in vals], fontsize=13)
    ax.axvline(0, color=C["muted"], lw=1)
    ax.axvline(0.766, color=C["line"], lw=1.5, ls="--", zorder=0)
    ax.text(0.766, y[0] + 0.55, "model card 0.766", ha="center", fontsize=11.5, color=C["muted"])
    ax.set_xlim(-1.25, 1.05)
    ax.set_ylim(-0.75, len(vals) - 0.2)
    ax.set_xlabel("reconstruction (FVE): share of the activation rebuilt from the text (higher = better)")
    ax.text(0.02, -0.5, "0 = predicting the mean", fontsize=11.5, color=C["muted"], va="center")
    style.save(
        fig, "bk_exp00_fve",
        title="The NLA reproduces its published reconstruction (FVE), and text-only explanations reconstruct far less",
        caption=("Reconstruction (fraction of variance explained, FVE) of the NLA's critic rebuilding the layer-42 activation from a text, on 64 natural activations "
                 "(k=4 samples each for the NLA, T=1, thinking off); bars are means, 95% bootstrap CIs over the 64 examples. "
                 "The shipped reader reproduces the card (0.761 vs 0.766); a text model writing from the source text gets 0.28 "
                 "and the source text itself 0.15 to 0.21. Caveat: the critic was co-trained on the NLA's register, so "
                 "off-register text is handicapped; the fair text ceiling lies somewhere between 0.28 and the warm start's 0.46."),
        section="backup:exp00", role="backup",
        sources=["experiments/00_nla_gate/results/fve_gate.json", "experiments/00_nla_gate/results/score_gate_textnla_lpcheck.json",
                 "experiments/00_nla_gate/results/offbyone.json"],
        checks=checks,
        notes=("FVE = 1 - mean per-element MSE / per-element variance of the rescaled gold set (baseline 0.4616). The TextNLA "
               "row uses the warm-start instruction (experiments/06_resolving_decomposition/data/nla_instruction.txt), not the "
               "NLA actor prompt."),
    )


if __name__ == "__main__":
    main()
