"""bk_exp12_recall: all 14 exp 12 concept rows, prompted AUROC per reader and free-readout stated/exhibited rates."""
from __future__ import annotations

import g2_common as g
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import C, save

ROWS = [("nj1_col", "line position (chars since newline)"), ("nj1_col_break", "line break coming"),
        ("bk_wordpos", "words since sentence start"), ("bk_listidx", "list item index (no numerals)"),
        ("bk_indent", "indentation level"), ("bk_quote", "inside a quotation"),
        ("nj3_predpos", "next word's part of speech"), ("agr_gender", "upcoming noun gender"),
        ("agr_number", "upcoming noun number"),
        ("nj2_tense", "tense"), ("bk_caps", "all caps vs sentence case"), ("ij_listnum", "'Item N' index"),
        ("ij2_lang", "language (French vs German)"), ("bk_code", "code language")]
GROUP_BREAKS = [6, 9]  # position | syntax | controls
CK = {"nla": "nla", "ol": "ol", "jlens": "jl"}


def main() -> None:
    n = len(ROWS)
    ypos = np.arange(n)[::-1].astype(float)
    for b in GROUP_BREAKS:
        ypos[b:] -= 0.6
    P, S, srcs = {}, {}, set()
    for f, _ in ROWS:
        for r in ("nla", "ol", "jlens"):
            P[(f, r)] = g.prompted(f, r, n_boot=1000)
            srcs.update(P[(f, r)]["sources"])
        for r in ("nla", "ol42"):
            S[(f, r)] = g.states_rate(f, r, n_boot=1000)
            srcs.add(S[(f, r)]["source"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 7.0), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.06})
    off = {"nla": 0.2, "ol": 0.0, "jlens": -0.2}
    for (f, lab), y0 in zip(ROWS, ypos):
        for r, name in (("nla", "NLA"), ("ol", "Oracle Lens"), ("jlens", "J-lens")):
            p = P[(f, r)]
            ax.plot(p["ci"], [y0 + off[r]] * 2, color=C[CK[r]], lw=1.8, alpha=0.5)
            ax.plot(p["reported"], y0 + off[r], "o", ms=7, color=C[CK[r]], mec="white", mew=0.8,
                    label=name if f == "nj1_col" else None)
    ax.axvline(0.5, color=C["line"], ls="--", lw=1.4, zorder=0)
    ax.axvline(0.8, color=C["line"], ls=":", lw=1.2, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=12.5)
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(ypos[-1] - 0.6, ypos[0] + 0.9)
    ax.text(0.5, ypos[0] + 0.55, "chance", ha="center", fontsize=11, color=C["muted"])
    ax.text(0.8, ypos[0] + 0.55, "0.80", ha="center", fontsize=11, color=C["muted"])
    ax.set_xlabel("Prompted AUROC (0.5 = chance)")
    ax.legend(loc="lower left", fontsize=11.5, handletextpad=0.2, borderaxespad=0.1)
    for gy, name in ((ypos[0] + 0.25, "position in text"), (ypos[6] + 0.45, "upcoming-word syntax"),
                     (ypos[9] + 0.45, "controls")):
        pass
    for b in GROUP_BREAKS:
        for a in (ax, bx):
            a.axhline((ypos[b - 1] + ypos[b]) / 2, color=C["line"], lw=1)

    for (f, lab), y0 in zip(ROWS, ypos):
        s, o = S[(f, "nla")], S[(f, "ol42")]
        bx.barh(y0 + 0.17, s["rate"], height=0.32, color=C["nla"])
        bx.plot(s["ci"], [y0 + 0.17] * 2, color=C["ink"], lw=1)
        bx.barh(y0 - 0.17, o["exhibit_rate"], height=0.32, color="none", edgecolor=C["ol"], hatch="///", lw=1)
        bx.barh(y0 - 0.17, o["rate"], height=0.32, color=C["ol"])
        if s["rate"] > 0.005:
            bx.text(max(s["ci"][1], s["rate"]) + 0.01, y0 + 0.17, f"{100 * s['rate']:.0f}%", va="center", fontsize=10.5, color=C["nla"])
        if o["exhibit_rate"] > 0.005:
            bx.text(o["exhibit_rate"] + 0.01, y0 - 0.17, f"{100 * o['exhibit_rate']:.0f}%", va="center", fontsize=10.5, color=C["ol"])
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_ylim(ax.get_ylim())
    bx.set_xlim(0, 1.05)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1])
    bx.set_xticklabels(["0", "25", "50", "75", "100"])
    bx.set_xlabel("Free readouts (%)")
    bx.legend(handles=[Patch(color=C["nla"], label="NLA states it"), Patch(color=C["ol"], label="Oracle Lens states it"),
                       Patch(facecolor="none", edgecolor=C["ol"], hatch="///", label="Oracle Lens exhibits it")],
              loc="center right", bbox_to_anchor=(1.0, 0.69), fontsize=11.5)

    nla_v = np.array([P[(f, "nla")]["reported"] for f, _ in ROWS])
    ol_v = np.array([P[(f, "ol")]["reported"] for f, _ in ROWS])
    ge = int((ol_v >= nla_v).sum())
    ge2 = int((np.round(ol_v, 2) >= np.round(nla_v, 2)).sum())
    ol_states = sum(S[(f, "ol42")]["k"] for f, _ in ROWS)
    ol_n = sum(S[(f, "ol42")]["n"] for f, _ in ROWS)
    rec = {"nj1_col": 0.60, "bk_wordpos": 0.72, "bk_indent": 0.87, "bk_quote": 0.83}
    checks = [g.check("rows with Oracle Lens >= NLA prompted (unrounded)", ge, 9, "exp 12 RESULTS point 4 ('OLens >= NLA on 9 of 14')", tol=0),
              g.check("rows with Oracle Lens >= NLA prompted (2-dp rounded)", ge2, 9, "exp 12 RESULTS point 4", tol=0),
              g.check("Oracle Lens 'states' verdicts at layer 42", f"{ol_states}/{ol_n}", "0/4480", "exp 12 RESULTS point 1")]
    checks += [g.check(f"{f} NLA prompted", round(P[(f, 'nla')]["reported"], 2), v, "exp 12 RESULTS point 3/4") for f, v in rec.items()]
    checks += [g.check("list item index NLA stated", round(S[("bk_listidx", "nla")]["rate"], 2), 0.25, "exp 12 RESULTS point 2"),
               g.check("code language Oracle Lens exhibits", round(S[("bk_code", "ol42")]["exhibit_rate"], 2), 0.93, "exp 12 RESULTS point 1"),
               g.check("caps Oracle Lens exhibits", round(S[("bk_caps", "ol42")]["exhibit_rate"], 2), 0.50, "exp 12 RESULTS point 1")]
    save(fig, "bk_exp12_recall",
         title="Oracle Lens states a concept in 1 of 4,480 readouts; it only exhibits language, code, caps and tense",
         caption=(f"Exp 12, layer 42. Left: prompted AUROC per reader (better of mean-of-3 mined phrasings and fixed label; 160 items; "
                  f"95% bootstrap CI clustered by source). Right: share of 320 free readouts (40 items x 8) judged to state the value "
                  f"(NLA, Oracle Lens) or, for Oracle Lens, whose own text exhibits it (hatched); NLA CI item-clustered bootstrap. "
                  f"Oracle Lens >= NLA prompted on {ge} of 14 rows (the record says 9). Oracle Lens states {ol_states} of {ol_n} readouts. "
                  "One judge (Gemini 3.8 Flash)."),
         section="backup:exp12", role="backup", sources=sorted(srcs), checks=checks,
         notes=("Recount of 'OLens >= NLA on 9 of 14' gives the number in checks; the record's statement is wrong. The one Oracle Lens "
                "'states' verdict is on language (1 of 320)."))


if __name__ == "__main__":
    main()
