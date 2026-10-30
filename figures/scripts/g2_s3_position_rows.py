"""s3_position_rows: prompted AUROC per reader and free-readout 'states it' rate, position rows + language control."""
from __future__ import annotations

import g2_common as g
import matplotlib.pyplot as plt
import numpy as np
import style
from style import C, save

ROWS = [("tb_tablecol", "table column"),
        ("tb_paren", "inside parentheses"),
        ("fmt_listdepth", "list nesting depth"),
        ("nj1_col_fresh", "line position"),
        ("ij2_lang", "language\n(control)")]
CK = {"nla": "nla", "ol": "ol", "jlens": "jl"}
READERS = [("nla", "NLA"), ("ol", "Oracle Lens"), ("jlens", "J-lens")]
# record values (talk script section 3 table; RESULTS.md of exps 12/13)
REC_P = {"tb_tablecol": (1.00, 0.74, 0.64), "tb_paren": (0.96, 0.73, 0.47), "fmt_listdepth": (0.96, 0.88, 0.96),
         "nj1_col_fresh": (0.51, 0.49, 0.53), "ij2_lang": (1.00, 1.00, 0.98)}
REC_S = {"tb_tablecol": (0.41, 0.0), "tb_paren": (0.30, 0.0), "fmt_listdepth": (0.04, 0.0), "ij2_lang": (0.81, 0.0)}


def main() -> None:
    ypos = np.array([4.6, 3.6, 2.6, 1.6, -0.15])
    P, S = {}, {}
    sources, checks = set(), []
    for f, _ in ROWS:
        for r, _ in READERS:
            p = g.prompted(f, r)
            P[(f, r)] = p
            sources.update(p["sources"])
        if f != "nj1_col_fresh":
            for r in ("nla", "ol42"):
                s = g.states_rate(f, r)
                S[(f, r)] = s
                sources.add(s["source"])

    style.slide_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.45), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.13})
    fig.subplots_adjust(left=0.215, right=0.94, bottom=0.14, top=0.91)
    off = {"nla": 0.22, "ol": 0.0, "jlens": -0.22}
    for f, _ in ROWS:
        y0 = ypos[[x for x, _ in ROWS].index(f)]
        for r, lab in READERS:
            p = P[(f, r)]
            y = y0 + off[r]
            ax.plot(p["ci"], [y, y], color=C[CK[r]], lw=2.2, alpha=0.6, solid_capstyle="butt")
            ax.plot(p["reported"], y, "o", ms=9, color=C[CK[r]], mec="white", mew=1.2, zorder=3,
                    label=lab if f == "tb_tablecol" else None)
    ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=0)
    ax.axvline(0.8, color=C["line"], lw=1.2, ls=":", zorder=0)
    ax.text(0.5, 4.95, "chance", ha="center", va="bottom", fontsize=15, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(0.8, 4.95, "0.80", ha="center", va="bottom", fontsize=15, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlim(0.35, 1.04)
    ax.set_ylim(-0.6, 5.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in ROWS], linespacing=1.0)
    ax.get_yticklabels()[-1].set_color(C["muted"])
    ax.axhline(0.55, color=C["line"], lw=1)
    ax.set_xlabel("AUROC (0.5 = chance)")
    ax.set_title("A. Asked: forced choice", loc="left", x=-0.3)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.215), ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)

    # right: states it
    for f, _ in ROWS:
        y0 = ypos[[x for x, _ in ROWS].index(f)]
        if f == "nj1_col_fresh":
            bx.text(0.02, y0, "no free-readout run", va="center", fontsize=15, color=C["muted"])
            continue
        for r, dy, col in (("nla", 0.17, C["nla"]), ("ol42", -0.17, C["ol"])):
            s = S[(f, r)]
            bx.barh(y0 + dy, s["rate"], height=0.3, color=col)
            lo, hi = s["ci"]
            bx.plot([lo, hi], [y0 + dy] * 2, color=C["ink"], lw=1.3)
            if f == "ij2_lang" and r == "ol42":
                ol_k, ol_n = s["k"], s["n"]
                continue
            txt = f"{100 * s['rate']:.0f}%"
            bx.text(max(hi, s["rate"]) + 0.015, y0 + dy, txt, va="center", fontsize=15,
                    color=col if r == "nla" else C["ol"])
        if f == "ij2_lang":
            s = S[(f, "ol42")]
            bx.barh(y0 - 0.17, s["exhibit_rate"], height=0.3, color="none", edgecolor=C["ol"], hatch="///", lw=1.2)
            bx.text(s["exhibit_rate"] + 0.015, y0 - 0.2, f"writes in it: {100 * s['exhibit_rate']:.0f}%",
                    va="center", fontsize=15, color=C["ol"], linespacing=1.05)
    bx.set_xlim(0, 1.0)
    bx.set_ylim(ax.get_ylim())
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.axhline(0.55, color=C["line"], lw=1)
    bx.set_xlabel("readouts stating it (%)")
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xticklabels(["0", "25", "50", "75", "100"])
    bx.set_title("B. Unprompted: free readout", loc="left")
    bx.text(0.33, 3.1, "Oracle Lens: 0 of 960\non these three rows", fontsize=15, color=C["ol"],
            va="center")
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(color=C["nla"], label="NLA"), Patch(color=C["ol"], label="Oracle Lens")],
              loc="center", bbox_to_anchor=(0.5, 0.215), ncol=2, handlelength=1.0, handletextpad=0.3, columnspacing=0.6, borderaxespad=0)

    # checks
    for f, _ in ROWS:
        for (r, _), rec in zip(READERS, REC_P[f]):
            checks.append(g.check(f"{f} prompted AUROC {r}", round(P[(f, r)]["reported"], 2), rec,
                                  "talk_script S3 table / exp 12-13 RESULTS"))
        if f in REC_S:
            checks.append(g.check(f"{f} NLA states rate", round(S[(f, 'nla')]["rate"], 2), REC_S[f][0], "talk_script S3 table"))
            checks.append(g.check(f"{f} Oracle Lens states rate", round(S[(f, 'ol42')]["rate"], 3), REC_S[f][1],
                                  "talk_script S3 table"))
    checks.append(g.check("line position fresh NLA, mean of 3 selected templates", round(P[("nj1_col_fresh", "nla")]["mean3"], 2),
                          0.48, "exp 13 RESULTS.md text ('NLA 0.48')", tol=0.005))
    checks.append(g.check("line position fresh NLA, reported (max of mean3, fixed)", round(P[("nj1_col_fresh", "nla")]["reported"], 2),
                          0.51, "exp 13 RESULTS.md table / talk script", tol=0.005))
    checks.append(g.check("language Oracle Lens exhibits rate", round(S[("ij2_lang", "ol42")]["exhibit_rate"], 2), 0.56,
                          "exp 12 RESULTS.md"))
    checks.append(g.check("language Oracle Lens states count", S[("ij2_lang", "ol42")]["k"], 0, "talk script '0%'"))

    which = {f"{f}/{r}": P[(f, r)]["which"] for f, _ in ROWS for r, _ in READERS}
    save(fig, "s3_position_rows",
         title="Asked, the NLA detects every readable position row; unprompted, only the NLA ever states one",
         caption=("Rows: table column (first vs 4th-6th cell), inside vs outside parentheses, list nesting depth (top vs third level), "
                  "line position on fresh items, and the language control (French vs German). Left: prompted AUROC per reader (log-odds between two frozen phrasings; the record's convention, the better of "
                  "the mean-of-3 mined phrasings and the fixed label), 160 held-out items per row, 95% bootstrap CI clustered by "
                  "source passage. B: share of free readouts (40 items x 8 samples = 320 per bar) that a judge reads as stating "
                  "the value; CI is an item-clustered bootstrap (Wilson on readouts where the count is 0). Oracle Lens states nothing "
                  "on the position rows and writes French/German itself on 56% of language readouts (hatched) while stating the language in 1 of 320. Caveat: one judge "
                  "(Gemini 3.8 Flash); line position on fresh items is at chance for every reader and had no free-readout run."),
         section="S3", role="main", sources=sorted(sources), checks=checks,
         notes=("Layer 42; NLA iter_000400, Oracle Lens s3d-rl600 at 42, J-lens n1000 at 42. Variant used per cell "
                f"(mean3 = mean of the 3 selected mined phrasings, fixed = fixed label): {which}. "
                "Line position fresh: NLA mean3 0.48 vs fixed 0.51; the table reports 0.51 (max), the RESULTS text 0.48. "
                "Language: Oracle Lens has 1 'states' verdict of 320 (0.3%), recorded as 0%. "
                "NLA actor prompt (notes/wsbench_internals.md): 'You are a meticulous AI researcher ... <concept>...</concept> "
                "Please provide an explanation.'; the prompted phrasing is scored as the continuation after '<explanation>\\n'."))


if __name__ == "__main__":
    main()
