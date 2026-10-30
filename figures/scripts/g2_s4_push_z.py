"""s4_push_z (exp 15): z against 20 random directions for model and readers, per pushed direction."""
from __future__ import annotations

import g2_common as g
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style
from style import C, save

sys_path = str(g.EXP / "17_harden")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
import ci_prior  # noqa: E402  (read-only import; recomputes from experiments/15_ar_steering/results/vol/*.pt)

DIRS = [("cm_raw", "class mean"), ("arspan_raw", "rebuilt from\nthe true span"), ("arstmt_raw", "rebuilt from\na statement")]
SERIES = [("model", "model's next token", "probe"), ("nla_mean", "NLA", "nla"), ("ol_mean", "Oracle Lens", "ol"),
          ("jlens", "J-lens", "jl")]
MODEL_KEY = {"tb_tablecol": ("model_lo_newline", 1), "tb_paren": ("model_lo_close", -1)}
TITLES = {"tb_tablecol": "table column", "tb_paren": "inside parentheses"}


def main() -> None:
    R = {row: ci_prior.ci15(row, ci_prior.VOL15, n_boot=2000, seed=17) for row in ("tb_tablecol", "tb_paren")}
    style.slide_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.15), sharey=True, gridspec_kw={"wspace": 0.07})
    fig.subplots_adjust(left=0.2, right=0.985, bottom=0.15, top=0.85)
    Z = {}
    offs = {"model": 0.33, "nla_mean": 0.11, "ol_mean": -0.11, "jlens": -0.33}
    for ax, row in zip(axes, R):
        for i, (d, _) in enumerate(DIRS):
            y0 = 2 - i
            cell = R[row]["cells"][f"{d}:r11"]
            for key, lab, ck in SERIES:
                k, sgn = (MODEL_KEY[row] if key == "model" else (key, 1))
                c = cell[k]
                z, zl, zh = sgn * c["z_iso"], sgn * c["z_lo_iso"], sgn * c["z_hi_iso"]
                zl, zh = min(zl, zh), max(zl, zh)
                Z[(row, d, key)] = (z, zl, zh)
                y = y0 + offs[key]
                big = key == "model"
                ax.plot([zl, zh], [y, y], color=C[ck], lw=2.5, alpha=0.6)
                ax.plot(z, y, "s" if big else "o", ms=11 if big else 8, color=C[ck], mec="white", mew=1, zorder=3,
                        label=lab if (i == 0 and ax is axes[0]) else None)
                if big or d == "arstmt_raw":
                    ax.text(max(z, zh) + 0.8, y, f"{z:+.1f}", va="center", fontsize=15, color=C[ck],
                            fontweight="bold" if big else None)
        ax.axvspan(-2, 2, color=C["line"], alpha=0.35, zorder=0)
        ax.axvline(0, color=C["muted"], lw=0.8, zorder=0)
        ax.axhspan(-0.48, 0.48, color=C["text_pale"], alpha=0.35, zorder=0)
        ax.set_xlim(-6, 31)
        ax.set_ylim(-0.58, 2.9)
        ax.set_title(TITLES[row], loc="left", pad=4)
        ax.text(0, 2.5, "|z|<2", ha="center", va="bottom", fontsize=15, color=C["muted"],
                bbox=dict(fc="white", ec="none", pad=0.3))
    axes[0].set_yticks([2, 1, 0])
    axes[0].set_yticklabels([lab for _, lab in DIRS], linespacing=1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.59, 1.02), ncol=4, handletextpad=0.1, columnspacing=0.9,
               borderaxespad=0.1)
    fig.text(0.59, 0.035, "z against 20 random directions of the same size (+ = expected direction)", ha="center",
             va="center", fontsize=15)
    zt = Z[("tb_tablecol", "arstmt_raw", "model")][0]
    axes[0].text(14.5, 0.86, "readers follow,\nmodel does not", fontsize=15, color=C["ink"],
                 va="center")

    rec = {("tb_tablecol", "cm_raw", "model"): 7.1, ("tb_paren", "cm_raw", "model"): 19.8,
           ("tb_tablecol", "arspan_raw", "model"): 10.5, ("tb_paren", "arspan_raw", "model"): 6.0,
           ("tb_tablecol", "arstmt_raw", "model"): 0.7, ("tb_paren", "arstmt_raw", "model"): -3.6,
           ("tb_tablecol", "arstmt_raw", "ol_mean"): 12.3, ("tb_tablecol", "arstmt_raw", "jlens"): 10.1,
           ("tb_tablecol", "arstmt_raw", "nla_mean"): 4.6, ("tb_paren", "arstmt_raw", "ol_mean"): 3.6,
           ("tb_paren", "arstmt_raw", "jlens"): -1.0, ("tb_paren", "arstmt_raw", "nla_mean"): 2.6,
           ("tb_tablecol", "cm_raw", "nla_mean"): 13.2, ("tb_paren", "cm_raw", "ol_mean"): 10.4}
    checks = [g.check(f"{r} {d} {k} z", round(Z[(r, d, k)][0], 1), v, "exp 15 RESULTS headline table", tol=0.051)
              for (r, d, k), v in rec.items()]
    checks.append(g.check("parenthesis span-rebuilt direction, model z", round(Z[("tb_paren", "arspan_raw", "model")][0], 1), 12.9,
                          "talk script S4 exp 15 table ('rebuilt from the true span ... +12.9')", tol=0.051))
    checks.append(g.check("tb_tablecol statement model z CI ends", [round(v, 1) for v in Z[("tb_tablecol", "arstmt_raw", "model")][1:]],
                          [0.4, 1.1], "exp 15 RESULTS CI section"))
    save(fig, "s4_push_z",
         title="A statement-built direction moves the readers, not the model",
         caption=("Exp 15, layer 42, 200 items per row. Each activation is pushed along a direction (centred edit at 11% of its norm) "
                  "and the paired shift in the model's next-token log-odds (newline for table column, ')' for parenthesis) and in each "
                  "reader's prompted log-odds is expressed as z against 20 random directions of the same size (grey band |z| < 2). "
                  "Bars: z of the 95% CI ends of the shift, cluster bootstrap by passage, 2,000 resamples. Directions are built by the "
                  "NLA's reconstructor, so the NLA's response to them is partly circular; the class-mean row is the clean comparison."),
         section="S4", role="main",
         sources=["experiments/15_ar_steering/results/vol/spec15_tb_tablecol.pt", "experiments/15_ar_steering/results/vol/spec15_tb_paren.pt",
                  "experiments/15_ar_steering/results/vol/behave15_tb_tablecol_c0.pt", "experiments/15_ar_steering/results/vol/behave15_tb_tablecol_c1.pt",
                  "experiments/15_ar_steering/results/vol/behave15_tb_paren_c0.pt", "experiments/15_ar_steering/results/vol/behave15_tb_paren_c1.pt",
                  "experiments/15_ar_steering/results/vol/score15_{nla,ol}_{tb_tablecol,tb_paren}.pt"],
         checks=checks,
         notes=("Recomputed with experiments/17_harden/ci_prior.ci15 (imported read-only, bytecode writing disabled) from the exp 15 .pt "
                "files. Parenthesis model z sign flipped so + = 'outside' lowers ')' log-odds, as in the RESULTS headline. The talk "
                "script's S4 table gives the span-rebuilt direction for parenthesis as +12.9; the raw data give +6.0 (12.9 is the "
                "AR(NLA explanations) row). Statement pairs: NLA fixed labels ('The current cell is in the first/fifth column of the "
                "table'; 'The current position is inside/outside a parenthesis')."))


if __name__ == "__main__":
    main()
