"""bk_exp05_planted: planting a bridge from L34 on flips the answer, and all readers report the plant (exp 05)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import style  # noqa: E402
from style import C, ROOT  # noqa: E402

R = ROOT / "experiments" / "05_planted_intermediates" / "results"
LAYERS = [24, 28, 34, 38, 42]
REC_KEEP = {24: 18, 28: 39, 34: 212, 38: 431, 42: 542}
REC_READ = {"OL@42": (0.01, 1.00), "NLA@42": (0.03, 0.98), "J-Lens@42": (0.01, 1.00)}


def main() -> None:
    y = json.loads((R / "yield.json").read_text())
    rd = json.loads((R / "readers.json").read_text())
    checks = []
    fig, (ax, bx) = plt.subplots(1, 2, figsize=style.FULL, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})
    for i, L in enumerate(LAYERS):
        c = y["cells"][f"clamp_L{L}"]
        k, n = c["minalpha_policy"]["n_kept"], c["n_candidates"]
        p, lo, hi = style.wilson(k, n)
        key = L == 38
        ax.bar(i, p, 0.62, color=C["probe"] if key else C["line"], zorder=2)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=C["ink"], capsize=3, lw=1.1, zorder=3)
        ax.text(i, hi + 0.015, f"{k}", ha="center", va="bottom", fontsize=13, fontweight="bold" if key else None)
        checks.append({"quantity": f"kept items L{L} clamp, min-alpha<=4", "computed": k, "record": REC_KEEP[L],
                       "where": "exp 05 RESULTS yield table", "match": k == REC_KEEP[L]})
    dr = y["cells"]["clamp_L38"]["minalpha_policy"]["delta_ratio_quantiles"]
    ax.set_xticks(range(len(LAYERS)), [f"{L}" + ("\n(used)" if L == 38 else "") for L in LAYERS])
    ax.set_xlabel("layer where the bridge is planted")
    ax.set_ylabel(f"share of {n} candidate items kept\n(answer flips, model intact)")
    ax.set_ylim(0, 1.0)
    ax.set_title("A. Planting works from layer 34 on", loc="left", fontsize=15)
    ax.text(-0.3, 0.9, f"dose at layer 38: median |delta| / |h| = {dr[1]:.2f}\n(about 10 within-class SD)", fontsize=11.5,
            color=C["muted"], va="top")

    readers = [("NLA@42", "NLA", "nla"), ("OL@42", "Oracle Lens", "ol"), ("J-Lens@42", "J-lens", "jl")]
    w = 0.36
    for i, (key, lab, col) in enumerate(readers):
        r = rd["readers"][key]
        nn = r["n"]
        for j, (fld, face) in enumerate((("p_reads_Y_natural", C[f"{col}_pale"]), ("p_reads_Y_planted", C[col]))):
            p = r[fld]
            pp, lo, hi = style.wilson(round(p * nn), nn)
            xi = i + (j - 0.5) * w
            bx.bar(xi, p, w * 0.92, color=face, zorder=2)
            bx.errorbar(xi, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], color=C["ink"], capsize=3, lw=1.1, zorder=3)
            bx.text(xi, hi + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=12.5, fontweight="bold" if j else None)
        rec = REC_READ[key]
        comp = (r["p_reads_Y_natural"], r["p_reads_Y_planted"])
        checks.append({"quantity": f"{key} P(reads Y) unplanted / planted", "computed": f"{comp[0]:.3f} / {comp[1]:.3f}",
                       "record": f"{rec[0]:.2f} / {rec[1]:.2f}", "where": "exp 05 RESULTS reader pass",
                       "match": all(abs(a - b) <= 0.0051 for a, b in zip(comp, rec))})
    bx.set_xticks(range(3), [r[1] for r in readers])
    bx.set_xlabel("reader at layer 42 (pale = unplanted, dark = planted)")
    bx.set_ylim(0, 1.12)
    bx.set_ylabel("reader reports the planted country Y")
    bx.set_title("B. Every reader reports the plant", loc="left", fontsize=15)
    style.save(
        fig, "bk_exp05_planted",
        title="A planted bridge flips the model's answer from layer 34 on, and all three readers report it",
        caption=("A: share of 614 two-hop country items (e.g. 'the capital of the country famous for ...') where swapping the "
                 "bridge country's class-mean projection at that layer (centred clamp, smallest dose up to 4x the class-mean "
                 "gap) flips the model's answer while leaving it intact; Wilson 95% CIs, kept counts on bars. B: on 200 kept layer-38 "
                 "items, share where each reader's layer-42 log-odds prefers the planted country Y over the text-implied X (mean of 3 "
                 "templates), Wilson 95%. Caveat: the swap direction is strongly J-enriched (6.5x) and doses are large, so this "
                 "is a calibration that reading works for in-J content, not a test of non-J reading."),
        section="backup:exp05", role="backup",
        sources=["experiments/05_planted_intermediates/results/yield.json",
                 "experiments/05_planted_intermediates/results/readers.json"],
        checks=checks,
        notes=("Reader templates: Oracle Lens '- {c}', '- {c} (country)', '- The country of {c}'; NLA '{c}', 'The text is about {c}', "
               "'A factual question involving {c}' after '<explanation>\\n'. RESULTS rounds Oracle Lens unplanted 0.005 to 0.01 and NLA "
               "planted 0.985 to 0.98."),
    )


if __name__ == "__main__":
    main()
