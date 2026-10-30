"""bk_exp04_backup: no backup candidate is both J-depleted and used (exp 04)."""
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

R = ROOT / "experiments" / "04_backup_features" / "results"
FEATS = [("indent", "indentation level"), ("sent_pos", "words since sentence start"),
         ("code_lang", "coding language (Python vs JS)"), ("cap_next_all", "next word capitalised (all)"),
         ("in_quote", "inside a quotation"), ("list_num", "list item index (mid-item)"), ("list_hdr", "list item index (header)"),
         ("cap_next", "next word capitalised (mid-sentence)"), ("gender_next", "French gender of upcoming noun")]
REC_ENR = {"cap_next": 2.71, "cap_next_all": 1.75, "code_lang": 1.49, "in_quote": 2.22, "gender_next": 4.15, "indent": 1.11,
           "sent_pos": 1.03, "list_num": 2.37, "list_hdr": 2.45}
REC_G4 = {"cap_next": (7.3, "p90"), "cap_next_all": (12.5, "p90"), "code_lang": (5.5, "p90"), "in_quote": (2.6, "p99"),
          "gender_next": (2.8, "p99"), "indent": (0.6, "p90"), "sent_pos": (3.7, "p99"), "list_num": (0.0, "p90"),
          "list_hdr": (0.4, "p90")}


def g4(feat: str) -> dict:
    """Signed ratio (effect / largest |random| effect) with CI, per dose."""
    g = json.loads((R / "causal" / f"{feat}_L38.json").read_text())["g4"]
    jcmax = g["jc_abs_max"]
    out = {}
    for d in ("p90", "p99"):
        c = g[f"jrem1536_{d}"]
        out[d] = (c["effect"] / jcmax, c["ci95"][0] / jcmax, c["ci95"][1] / jcmax)
    return out


def main() -> None:
    gates = json.loads((R / "gates.json").read_text())
    checks = []
    fig, (ax, bx) = plt.subplots(1, 2, figsize=style.FULL, sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(FEATS))[::-1]
    for yi, (k, lab) in zip(y, FEATS):
        e, sh = gates[k]["enrich"]["384"], gates[k]["enrich_shuffled"]["384"]
        dep = e <= 1.5
        ax.scatter(sh, yi, marker="|", s=260, color=C["muted"], lw=2, zorder=2)
        ax.scatter(e, yi, s=120, color=C["probe"] if dep else C["line"], edgecolor=C["probe"], lw=1.5, zorder=3)
        ax.text(max(e, sh) + 0.14, yi, f"{e:.2f}", va="center", ha="left", fontsize=11.5,
                color=C["probe"] if dep else C["muted"], fontweight="bold" if dep else None)
        gg = g4(k)
        (r0, l0, h0), (r1, l1, h1) = gg["p90"], gg["p99"]
        bx.plot([r0, r1], [yi, yi], color=C["line"], lw=2, zorder=1)
        for (r, lo, hi), filled in ((gg["p90"], False), (gg["p99"], True)):
            used = r >= 5
            bx.errorbar(r, yi, xerr=[[r - lo], [hi - r]], color=C["ink"], capsize=2, lw=0.9, zorder=2)
            bx.scatter(r, yi, s=95, zorder=3, facecolor=(C["probe"] if used else C["muted"]) if filled else "white",
                       edgecolor=C["probe"] if used else C["muted"], lw=1.8)
        bx.text(max(h0, h1) + 0.4, yi, f"{r0:.1f}x / {r1:.1f}x", va="center", fontsize=11, color=C["ink"])
        checks.append({"quantity": f"{k} enrichment r*", "computed": round(e, 2), "record": REC_ENR[k],
                       "where": "exp 04 RESULTS gate table", "match": round(e, 2) == REC_ENR[k]})
        rv, rd = REC_G4[k]
        comp = abs(gg[rd][0])
        checks.append({"quantity": f"{k} G4 @L38 x max |random| at the dose RESULTS quotes ({rd})", "computed": round(comp, 1),
                       "record": rv, "where": "exp 04 RESULTS gate table ('best dose' column)", "match": round(comp, 1) == rv})
    ax.axvline(1.5, color=C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(1.45, y[0] + 0.75, "1.5", ha="right", fontsize=11.5, color=C["muted"])
    ax.set_xlim(0.5, 5.2)
    ax.set_ylim(-0.8, len(FEATS) - 0.1)
    ax.set_yticks(y, [f[1] for f in FEATS], fontsize=12.5)
    ax.set_xlabel("J-space enrichment at r*\n(at or below 1.5 = depleted)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["probe"], ms=9, label="feature"),
                       Line2D([], [], marker="|", ls="", color=C["muted"], ms=14, mew=2, label="shuffled-label reference")],
              loc="upper right", fontsize=11.5)
    ax.set_title("A. Outside the J-space?", loc="left", fontsize=15)
    bx.axvline(5, color=C["line"], ls="--", lw=1.5, zorder=0)
    bx.text(5.1, y[0] + 0.75, "5x = used", fontsize=11.5, color=C["muted"])
    bx.set_xlim(-2.5, 21)
    bx.axvline(0, color=C["muted"], lw=1, zorder=0)
    bx.set_xlabel("next-token effect of the J-removed push\n(multiple of the largest random direction)")
    bx.set_title("B. Used by the model?", loc="left", fontsize=15)
    bx.tick_params(axis="y", length=0)
    bx.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=C["muted"], mew=1.8, ms=9, label="q90 dose"),
                       Line2D([], [], marker="o", ls="", color=C["muted"], ms=9, label="q99 dose")],
              loc="lower right", fontsize=11.5)
    bx.text(9.5, y[0], "depleted but unused:\nmoves the wrong way", va="center", fontsize=11.5, color=C["probe"])
    style.save(
        fig, "bk_exp04_backup",
        title="No backup candidate is both outside the J-space and used; the column count stays the only clean one",
        caption=("Nine backup non-J candidates at layer 42. A: enrichment of the class-mean direction in the top-384 J-space "
                 "(dark = at or below 1.5), with each feature's shuffled-label reference (tick). B: effect on the predicted "
                 "next-token behaviour of steering at layer 38 along the direction with the top-1536 J-space removed, as a multiple "
                 "of the largest |effect| of 24 random J-complement directions, signed, at the q90 (open) and q99 (filled) doses "
                 "(error bar: 95% item-bootstrap CI of the effect divided by that random maximum). Only indentation and sentence position are depleted; "
                 "indentation does not move, sentence position is J-lens readable. Caveat: point estimates in A carry no CI."),
        section="backup:exp04", role="backup",
        sources=["experiments/04_backup_features/results/gates.json",
                 *[f"experiments/04_backup_features/results/causal/{k}_L38.json" for k, _ in FEATS]],
        checks=checks,
        notes=("The RESULTS 'best dose' column is not the larger dose throughout: it quotes p90 for cap_next (7.3x; p99 is "
               "10.8x), cap_next_all (12.5x; p99 16.0x), code_lang (5.5x; p99 10.0x), indent and list_hdr, and p99 for in_quote, "
               "gender_next, sent_pos. gates.json G4_L38 stores p90 for all of them. Both doses are plotted. No verdict changes."),
    )


if __name__ == "__main__":
    main()
