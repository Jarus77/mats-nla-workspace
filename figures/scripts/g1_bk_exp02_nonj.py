"""bk_exp02_nonj: only the column count sits outside the J-space; and the r* choice (exp 02)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import style  # noqa: E402
from style import C, ROOT  # noqa: E402

G = ROOT / "experiments" / "02_nonj_features" / "gates.json"
FEATS = [("col", "characters since last newline"), ("col_report", "same count, asked in a chat frame"),
         ("depth", "bracket depth"), ("nexttok", "next token"), ("pred_pos", "predicted part of speech"),
         ("lang", "language"), ("tense", "verb tense"), ("topic", "topic")]
REC = {"col": (0.67, 1.10), "col_report": (0.59, 0.86), "depth": (0.57, 1.16), "pred_pos": (2.62, 1.56),
       "tense": (5.60, 2.31), "lang": (3.32, 1.73), "topic": (5.98, 2.34), "nexttok": (1.29, 1.54)}
# r* curve: no raw file locally (rstar.json lives only on the Modal results volume); values from RESULTS.md
RSTAR_RANKS = [256, 384, 1024, 1536]
RSTAR_AGREE = [0.89, 0.93, 0.95, 0.97]


def main() -> None:
    g = json.loads(G.read_text())
    checks = []
    fig, (ax, bx) = plt.subplots(1, 2, figsize=style.FULL, gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.35})
    y = np.arange(len(FEATS))[::-1]
    for yi, (k, lab) in zip(y, FEATS):
        e = g[k]["G2_geometric_enrichment_diffmeans"]
        a, b = e["r384"], e["r1536"]
        key = k in ("col", "col_report")
        col = C["probe"] if key else C["muted"]
        ax.plot([a, b], [yi, yi], color=C["line"], lw=2, zorder=1)
        ax.scatter(a, yi, s=110, color=col, zorder=3)
        ax.scatter(b, yi, s=110, facecolor="white", edgecolor=col, lw=2, zorder=3)
        extra = "   (J-lens reads it via tokens: excluded)" if k == "depth" else ""
        ax.text(max(a, b) + 0.15, yi, f"{a:.2f} / {b:.2f}{extra}", va="center", fontsize=11.5, color=col,
                fontweight="bold" if key else None)
        checks.append({"quantity": f"{k} enrichment r*/4r*", "computed": f"{a:.2f}/{b:.2f}", "record": f"{REC[k][0]:.2f}/{REC[k][1]:.2f}",
                       "where": "exp 02 RESULTS gate table", "match": (round(a, 2), round(b, 2)) == REC[k]})
    ax.set_yticks(y, [f[1] for f in FEATS], fontsize=12.5)
    ax.axvline(1.5, color=C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(1.55, y[0] + 0.75, "1.5 = J-enriched", ha="left", fontsize=11.5, color=C["muted"])
    ax.axvline(1.0, color=C["line"], ls=":", lw=1.2, zorder=0)
    ax.text(1.0, y[-1] - 0.85, "1 = random", ha="center", fontsize=11, color=C["muted"])
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-1.1, len(FEATS) + 0.3)
    ax.set_xlabel("share of the feature direction inside the J-space,\nrelative to a random direction")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["muted"], ms=9, label="rank r* = 384"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=C["muted"], mew=2, ms=9, label="rank 4r* = 1536")],
              loc="upper right", fontsize=12)
    ax.set_title("A. Feature directions vs the J-space (layer 42)", loc="left", fontsize=15)

    bx.plot(RSTAR_RANKS, RSTAR_AGREE, "-o", color=C["jl"], lw=2.2, ms=8)
    for r, a_ in zip(RSTAR_RANKS, RSTAR_AGREE):
        bx.text(r, a_ + 0.006, f"{a_:.2f}", ha="center", fontsize=12)
    bx.axhline(0.9, color=C["line"], ls="--", lw=1.5, zorder=0)
    bx.text(1650, 0.902, "0.90", va="bottom", ha="right", fontsize=11.5, color=C["muted"])
    bx.annotate("r* = 384", xy=(384, 0.93), xytext=(650, 0.885), fontsize=13, color=C["jl"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["jl"]))
    bx.set_xlabel("rank of the rebuilt J-lens")
    bx.set_ylabel("top-10 agreement with the full J-lens")
    bx.set_ylim(0.86, 0.99)
    bx.set_xlim(150, 1700)
    bx.set_title("B. Choosing the J-space rank", loc="left", fontsize=15)
    checks.append({"quantity": "r* at L42", "computed": g["_meta"]["r_star"], "record": 384, "where": "exp 02 RESULTS / gates.json _meta",
                   "match": g["_meta"]["r_star"] == 384})
    checks.append({"quantity": "r* agreement curve at L42", "computed": "not computable: rstar.json absent locally",
                   "record": "256 0.89, 384 0.93, 1024 0.95, 1536 0.97", "where": "exp 02 RESULTS r* paragraph", "match": None})
    style.save(
        fig, "bk_exp02_nonj",
        title="Of eight natural features, only the column count is cleanly outside the J-space",
        caption=("A: J-space enrichment of each feature's class-mean direction at layer 42 (neuronpedia n1000 lens; J-space = top-r "
                 "right singular vectors of W_U diag(1+w) J_42); filled = r*, open = 4r*; at or below 1.5 counts as J-depleted. "
                 "Point estimates only (gates.json stores no CI for this statistic). The count passes, but the chat-frame "
                 "version meant as an in-J calibration also stays out. B: r* is the smallest rank whose rebuilt lens keeps "
                 "at least 90% top-10 agreement on 100 basic_readout items; values from RESULTS.md (no raw file locally), no CIs."),
        section="backup:exp02", role="backup", sources=["experiments/02_nonj_features/gates.json",
                                                         "experiments/02_nonj_features/RESULTS.md (panel B only)"],
        checks=checks,
        notes=("Panel B: rstar.json is written by s8_rstar.py to the Modal results volume and is not in the repo; per brief, "
               "plotted from RESULTS.md text with no error bars. Only layer-42 agreement-by-rank is recorded in RESULTS (r* for other "
               "layers: 38: 192, 40: 384, 41: 384, 44: 256), so a per-layer curve could not be drawn."),
    )


if __name__ == "__main__":
    main()
