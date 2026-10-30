"""s4_swap (exp 16): does the NLA's answer follow the donor through the remainder or through the Oracle Lens part?"""
from __future__ import annotations

import sys

import g2_common as g
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style
from style import C, save

sys.path.insert(0, str(g.EXP / "17_harden"))
import ci_prior  # noqa: E402  (read-only; recomputes from experiments/16_difference_term/results/vol/*.pt)

ROWS = [r for r, _ in g.S4_ROWS]


def main() -> None:
    R = {r: ci_prior.ci16(r, ci_prior.VOL16, n_boot=2000, seed=17) for r in ROWS}
    style.slide_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.25))
    fig.subplots_adjust(left=0.2, right=0.975, bottom=0.17, top=0.86)
    V = {}
    for r in ROWS:
        y = g.S4_Y[r]
        nla = R[r]["readers"]["nla"]
        res, olp = nla["swap_res"]["mean"], nla["swap_ol"]["mean"]
        V[r] = (res, olp, R[r]["readers"]["ol"])
        ax.plot([olp["point"], res["point"]], [y, y], color=C["line"], lw=3, zorder=1)
        for c, col, dy in ((res, C["probe"], 0.0), (olp, C["ol"], 0.0)):
            ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=6, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.plot(c["point"], y, "o", ms=14, color=col, mec="white", mew=1.5, zorder=3)
        for c, col in ((res, C["probe"]), (olp, C["ol"])):
            ax.text(c["point"], y + 0.24, f"{c['point']:.2f}", ha="center", fontsize=15, color=col, fontweight="bold")
    g.s4_rows(ax)
    ax.axvline(0.5, color=C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(0.5, -0.33, "chance", ha="center", va="bottom", fontsize=15, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("NLA's answer vs the donor's class (AUROC; 1 = follows the donor)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=13, color=C["probe"], label="donor supplies the remainder"),
                       Line2D([], [], marker="o", ls="", ms=13, color=C["ol"], label="donor supplies the Oracle Lens part")],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, handletextpad=0.1, columnspacing=1.0, borderaxespad=0.2)
    ax.text(0.5, 0.75, "the control flips", ha="center", fontsize=15, color=C["muted"], style="italic",
            bbox=dict(fc="white", ec="none", pad=0.5))

    rec = {"tb_tablecol": (0.98, 0.01), "tb_paren": (0.87, 0.12), "bk_listidx": (0.98, 0.03), "ij2_lang": (0.10, 0.92)}
    recci = {"tb_tablecol": ((0.96, 1.00), (0.00, 0.03)), "tb_paren": ((0.82, 0.92), (0.07, 0.17)),
             "bk_listidx": ((0.96, 0.99), (0.01, 0.05)), "ij2_lang": ((0.06, 0.15), (0.88, 0.95))}
    checks = []
    for r in ROWS:
        res, olp, _ = V[r]
        checks += [g.check(f"{r} NLA follows donor via remainder", round(res["point"], 2), rec[r][0], "exp 16 RESULTS / talk S4"),
                   g.check(f"{r} NLA follows donor via Oracle Lens part", round(olp["point"], 2), rec[r][1], "exp 16 RESULTS / talk S4"),
                   g.check(f"{r} remainder CI", [round(res["lo"], 2), round(res["hi"], 2)], list(recci[r][0]), "exp 16 RESULTS CI section"),
                   g.check(f"{r} Oracle Lens-part CI", [round(olp["lo"], 2), round(olp["hi"], 2)], list(recci[r][1]), "exp 16 RESULTS CI section")]
    widths = max(max(V[r][0]["hi"] - V[r][0]["point"], V[r][0]["point"] - V[r][0]["lo"]) for r in ROWS)
    checks.append(g.check("max CI half-width (talk: 'bootstrap intervals within 0.04')", round(widths, 2), 0.04, "talk script S4", tol=0.0))
    olr = {r: (round(V[r][2]["swap_res"]["mean"]["point"], 2), round(V[r][2]["swap_ol"]["mean"]["point"], 2)) for r in ROWS}
    save(fig, "s4_swap",
         title="The NLA reads position from the part of the vector Oracle Lens's words do not rebuild",
         caption=("Exp 16, layer 42, 200 items per row. Each activation is split into the part the NLA's reconstructor rebuilds from "
                  "Oracle Lens's bullets (the Oracle Lens part) and the remainder (what those words do not rebuild); one part is swapped in from a donor item of the opposite class. Dots: AUROC "
                  "of the NLA's prompted log-odds (mean of 4 phrasings) against the donor's label; bands: 95% bootstrap CI clustered by "
                  "passage (2,000 resamples). Position rows follow the remainder; the language control follows the Oracle Lens part. "
                  "Caveat: both parts are built with the NLA's reconstructor; one seeded donor pairing."),
         section="S4", role="main",
         sources=[f"experiments/16_difference_term/results/vol/s1_{r}.pt" for r in ROWS]
                 + [f"experiments/16_difference_term/results/vol/s2_{r}.pt" for r in ROWS],
         checks=checks,
         notes=(f"Recomputed with experiments/17_harden/ci_prior.ci16 (read-only import). Oracle Lens's own readout (remainder, Oracle Lens part): {olr}. "
                "Fixed-label phrasing agrees in RESULTS. The talk's 'intervals within 0.04' holds for table column and list item index; parenthesis "
                "and language intervals reach about +-0.05."))


if __name__ == "__main__":
    main()
