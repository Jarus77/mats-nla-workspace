"""s1_best_of_layers: Oracle Lens's published lead is a best-of-11-layers effect (exp 01)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import style  # noqa: E402
from style import C  # noqa: E402
from g1_wsb import E01, FAMILIES, FAMILY_LABEL, load_rows, rel  # noqa: E402

GRID = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
REC_ANY = {"typo_mt": 0.86, "multilingual_mt": 0.65, "multihop_mt": 0.34, "basic_readout_mt": 0.73}
REC_BEST = {"typo_mt": (0.78, 24), "multilingual_mt": (0.43, 56), "multihop_mt": (0.18, 56), "basic_readout_mt": (0.62, 56)}
REC_NLA = {"typo_mt": 0.73, "multilingual_mt": 0.42, "multihop_mt": 0.15, "basic_readout_mt": 0.57}
REC_OL42 = {"typo_mt": 0.76, "multilingual_mt": 0.28, "multihop_mt": 0.10, "basic_readout_mt": 0.45}


def main() -> None:
    grid, nla, ol42 = load_rows(E01, "olens_grid"), load_rows(E01, "nla_L42"), load_rows(E01, "olens_L42")
    res, checks = {}, []
    for fam in FAMILIES:
        keys = [k for k in grid if k[0] == fam]
        n = len(keys)
        per_layer = {L: sum(L in grid[k]["passing_layers"] for k in keys) for L in GRID}
        bestL = max(GRID, key=lambda L: (per_layer[L], -GRID.index(L) * 0))
        best_k = per_layer[bestL]
        any_k = sum(bool(grid[k]["passing_layers"]) for k in keys)
        nla_k = sum(nla[k]["pass"] for k in keys)
        ol_k = sum(ol42[k]["pass"] for k in keys)
        res[fam] = dict(n=n, any=any_k, best=best_k, bestL=bestL, nla=nla_k, ol42=ol_k)
        checks += [
            {"quantity": f"{fam} Oracle Lens any-of-11", "computed": any_k / n, "record": REC_ANY[fam], "where": "exp 01 RESULTS replication table",
             "match": round(any_k / n, 2) == REC_ANY[fam]},
            {"quantity": f"{fam} Oracle Lens best single layer", "computed": f"{best_k / n:.2f} at layer {bestL}", "record": f"{REC_BEST[fam][0]} at layer {REC_BEST[fam][1]}",
             "where": "exp 01 best_of_layers_check.md", "match": round(best_k / n, 2) == REC_BEST[fam][0] and bestL == REC_BEST[fam][1]},
            {"quantity": f"{fam} NLA@42", "computed": nla_k / n, "record": REC_NLA[fam], "where": "exp 01 RESULTS", "match": round(nla_k / n, 2) == REC_NLA[fam]},
            {"quantity": f"{fam} Oracle Lens at layer 42", "computed": ol_k / n, "record": REC_OL42[fam], "where": "exp 01 RESULTS", "match": round(ol_k / n, 2) == REC_OL42[fam]},
        ]
    gaps = [(res[f]["any"] - res[f]["best"]) / res[f]["n"] for f in FAMILIES]
    checks.append({"quantity": "any-of-11 minus best single layer, range", "computed": f"+{min(gaps):.2f} to +{max(gaps):.2f}",
                   "record": "+0.08 to +0.22", "where": "talk script S1", "match": round(min(gaps), 2) == 0.08 and round(max(gaps), 2) == 0.22})

    series = [("nla", "NLA, layer 42", C["nla"], None), ("ol42", "Oracle Lens, layer 42", C["ol"], None),
              ("best", "Oracle Lens, best single layer", C["ol_pale"], None),
              ("any", "Oracle Lens, best of 11 layers per item (published)", "white", C["ol"])]
    style.slide_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.45))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.10, top=0.82)
    w = 0.235
    x = np.arange(len(FAMILIES))
    for j, (key, lab, face, edge) in enumerate(series):
        for i, fam in enumerate(FAMILIES):
            r = res[fam]
            p, lo, hi = style.wilson(r[key], r["n"])
            xi = i + (j - 1.5) * w
            ax.bar(xi, p, w * 0.92, color=face, edgecolor=edge or face, hatch="//" if edge else None, lw=1.5 if edge else 0,
                   zorder=2, label=lab if i == 0 else None)
            ax.errorbar(xi, p, yerr=[[p - lo], [hi - p]], color=C["ink"], capsize=2.5, lw=1.0, zorder=3)
            # values run vertically inside the bar (outside for short bars) so 16 bars stay legible at 15 pt
            inside = p > 0.3
            if key == "any":  # published score and its best-of-11 gain sit above the hatched bar
                ax.text(xi, hi + 0.015, f"+{(r['any'] - r['best']) / r['n']:.2f}\n{p:.2f}", ha="center", va="bottom",
                        fontsize=15, color=C["ol"], fontweight="bold", linespacing=1.05)
                continue
            tc = "white" if inside and key in ("nla", "ol42") else C["ink"]
            ax.text(xi, 0.025 if inside else hi + 0.02, f"{p:.2f}", rotation=90, ha="center", va="bottom", fontsize=15,
                    color=tc, fontweight="bold" if key in ("nla", "ol42") else None, zorder=4)
    ax.set_xticks(x, [FAMILY_LABEL[f] for f in FAMILIES])
    ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("pass rate (higher = better)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.09, 1.0), ncol=2, handlelength=1.4, columnspacing=1.0,
              borderaxespad=0.1, labelspacing=0.3)
    ax.text(2.0, 1.07, "purple: published score, and\nits gain over the best single layer", ha="center", va="top", fontsize=15,
            color=C["ol"], linespacing=1.1)
    style.save(
        fig, "s1_best_of_layers",
        title="Oracle Lens's published lead comes from picking the best of 11 layers per item",
        caption=("Regex pass rate per WorkspaceBench family (n=100 items each, k=1 sample per cell, Wilson 95% CIs). "
                 "At the matched layer 42 the NLA ties Oracle Lens on typo and beats it on the other three; the published "
                 "Oracle Lens score counts an item as passed if any of 11 layers (20 to 60, step 4) passes, which adds "
                 f"+{min(gaps):.2f} to +{max(gaps):.2f} over Oracle Lens's best single layer per family (numbers above). "
                 "Caveat: any-of-11 is also a best-of-11-samples bonus, so part of it is sampling, not layer choice; "
                 "layer 42 is off Oracle Lens's training grid but matches its on-grid neighbours, layers 40 and 44. "
                 "Best single layers: " + ", ".join(f"{FAMILY_LABEL[f]} layer {res[f]['bestL']}" for f in FAMILIES) + "."),
        section="S1", role="main",
        sources=[rel(E01 / "judged" / a) + "/*/results.json" for a in ("olens_grid", "olens_L42", "nla_L42")],
        checks=checks,
        notes=("Best single layer is chosen per family after the fact (it favours Oracle Lens). Reader prompts: NLA actor "
               "prompt per notes/wsbench_internals.md; Oracle Lens carrier 'An activation vector from layer L ... Produce "
               "distinct concepts that encode this activation, each as a '- ' bullet on its own line.'"),
    )


if __name__ == "__main__":
    main()
