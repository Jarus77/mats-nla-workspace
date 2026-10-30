"""Exp 17 backups: bk_exp17_paren_fourway (four-way re-score of parenthesis) and bk_exp17_cross_scoring
(independent bank x phrasings)."""
from __future__ import annotations

import g2_common as g
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.special import logsumexp
from style import C, FULL, auroc, bootstrap_ci, save

VOL = g.EXP / "17_harden" / "results" / "vol"


def paren_fourway() -> None:
    nla = torch.load(VOL / "mc_nla_tb_paren.pt", weights_only=False, map_location="cpu")
    ol = torch.load(VOL / "mc_ol_tb_paren.pt", weights_only=False, map_location="cpu")
    lp = nla["lp"]["nla:mc:state"].numpy().astype(float)
    y = np.asarray(nla["y"]).astype(int)  # 0 = inside, 1 = outside
    truth = np.asarray(nla["truth"])
    cl = np.asarray(nla["clusters"])
    am = lp.argmax(1)
    cin, cout = np.bincount(am[y == 0], minlength=4), np.bincount(am[y == 1], minlength=4)
    hit = (am == truth).astype(float)
    t1, t1ci = hit.mean(), bootstrap_ci(lambda a: float(a.mean()), hit, groups=cl)

    def cauc(L, yy):
        return auroc(L[:, 1] - L[:, 0], yy == 1)  # spec17 CLASS_SETS: inside [0], outside [1]

    cells = [("NLA", "nla", lp, y, cl), ("Oracle Lens", "ol", ol["lp"]["ol:mc:state"].numpy().astype(float),
                                          np.asarray(ol["y"]).astype(int), np.asarray(ol["clusters"])),
             ("J-lens", "jl", ol["lp"]["jlens:mc:state"].numpy().astype(float), np.asarray(ol["y"]).astype(int),
              np.asarray(ol["clusters"]))]
    res = [(n, ck, cauc(L, yy), bootstrap_ci(cauc, L, yy, groups=c)) for n, ck, L, yy, c in cells]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=FULL, gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
    names = ["inside a\nparenthesis", "outside any\nparenthesis", "inside a\nquotation", "inside square\nbrackets"]
    x = np.arange(4)
    w = 0.38
    b1 = ax.bar(x - w / 2, cin, w, color=C["nla"], label=f"true inside (n = {int((y == 0).sum())})")
    b2 = ax.bar(x + w / 2, cout, w, color=C["nla_pale"], label=f"true outside (n = {int((y == 1).sum())})")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{int(b.get_height())}", ha="center", fontsize=12.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12.5)
    ax.set_ylabel("items")
    ax.set_title("NLA's top choice of four ('The current position is ...')", loc="left", fontsize=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.annotate("never 'outside'", xy=(1 + w / 2, 1), xytext=(1.3, 35), fontsize=12.5, color=C["muted"],
                arrowprops=dict(arrowstyle="-", color=C["muted"]))
    yp = np.arange(3)[::-1]
    for (n, ck, a, ci), yy in zip(res, yp):
        bx.plot(ci, [yy, yy], color=C[ck], lw=3, alpha=0.55)
        bx.plot(a, yy, "o", ms=12, color=C[ck], mec="white")
        bx.text(a, yy + 0.22, f"{a:.2f}", ha="center", fontsize=12.5, color=C[ck])
    bx.axvline(0.5, color=C["line"], ls="--", lw=1.4, zorder=0)
    bx.set_yticks(yp)
    bx.set_yticklabels([r[0] for r in res])
    bx.set_xlim(0.25, 1.0)
    bx.set_ylim(-0.6, 2.6)
    bx.set_xlabel("Inside vs outside AUROC\n(four candidates; 0.5 = chance)", fontsize=13)
    bx.set_title("Discrimination survives", loc="left", fontsize=14)
    checks = [g.check("NLA argmax inside items", cin.tolist(), [61, 0, 12, 0], "exp 17 mc17.md"),
              g.check("NLA argmax outside items", cout.tolist(), [27, 0, 49, 0], "exp 17 mc17.md"),
              g.check("NLA four-way top-1", round(t1, 2), 0.41, "exp 17 RESULTS (a)"),
              g.check("NLA four-way top-1 CI lower bound", round(t1ci[0], 2), 0.33, "exp 17 RESULTS (a)"),
              g.check("NLA class AUROC", round(res[0][2], 2), 0.88, "exp 17 RESULTS (a)"),
              g.check("Oracle Lens class AUROC", round(res[1][2], 2), 0.40, "exp 17 RESULTS (a)"),
              g.check("J-lens class AUROC", round(res[2][2], 2), 0.45, "exp 17 RESULTS (a)"),
              g.check("n items after exclusion", len(y), 149, "exp 17 mc17.md")]
    save(fig, "bk_exp17_paren_fourway",
         title="With four candidates the NLA still separates inside from outside but never picks 'outside'",
         caption=(f"Exp 17 (a), parenthesis items re-scored over four states (149 exp 13 selection items; 11 with an open quote or bracket "
                  f"excluded). Left: the NLA's argmax; outside items go to 'inside a quotation' or 'inside a parenthesis'. Four-way top-1 "
                  f"{t1:.2f} [{t1ci[0]:.2f}, {t1ci[1]:.2f}] (chance 0.25), below the declared 0.50 bar, so the row was ruled a "
                  "two-candidate artefact. Right: inside-vs-outside AUROC from the two class candidates, 95% bootstrap CI clustered "
                  "by passage. Caveat: raw argmax under a strong phrase prior; prior-centred, outside items pick 'outside' 34 of 76."),
         section="backup:exp17", role="backup",
         sources=[g.rel(VOL / "mc_nla_tb_paren.pt"), g.rel(VOL / "mc_ol_tb_paren.pt"), "experiments/17_harden/spec17.py"],
         checks=checks, notes="Oracle Lens phrasing '- {inside parentheses|outside parentheses|inside quotes|inside brackets}'.")


def cross_scoring() -> None:
    CELLS = [("his_x_his", "independent items,\nindependent phrasings", C["muted"]),
             ("his_x_ours", "independent items,\nour phrasings", C["probe"]),
             ("ours_x_his", "our items,\nindependent phrasings", C["text"])]
    rec = {("tb_tablecol", "nla"): (0.56, 0.85, 0.64), ("tb_tablecol", "ol"): (0.58, 0.72, 0.36),
           ("tb_paren", "nla"): (0.75, 0.85, 0.83), ("tb_paren", "ol"): (0.53, 0.78, 0.50)}
    fig, axes = plt.subplots(1, 2, figsize=FULL, sharex=True, gridspec_kw={"wspace": 0.3})
    checks, srcs = [], []
    for ax, row, title in zip(axes, ("tb_tablecol", "tb_paren"), ("table column", "parenthesis")):
        for j, (reader, rl) in enumerate((("nla", "NLA"), ("ol", "Oracle Lens"))):
            p = VOL / f"xs_{reader}_{row}.pt"
            srcs.append(g.rel(p))
            s = torch.load(p, weights_only=False, map_location="cpu")
            y0 = 1 - j
            for k, (cell, lab, col) in enumerate(CELLS):
                lo = s["lo"][cell]
                his = cell.startswith("his")
                yy = np.asarray(s["his_y"] if his else s["our_y"], int)
                cl = np.asarray(s["his_clusters"] if his else s["our_clusters"])
                sc = np.mean([np.asarray(v, float) for v in lo.values()], 0)
                a = auroc(sc, yy)
                ci = bootstrap_ci(auroc, sc, yy, groups=cl)
                y = y0 + 0.22 - 0.22 * k
                ax.plot(ci, [y, y], color=col, lw=2.5, alpha=0.5)
                ax.plot(a, y, "o", ms=10, color=col, mec="white", label=lab if (j == 0 and row == "tb_tablecol") else None)
                ax.text(ci[1] + 0.015, y, f"{a:.2f}", va="center", fontsize=11.5, color=col)
                checks.append(g.check(f"{row} {reader} {cell}", round(a, 2), rec[(row, reader)][k], "exp 17 RESULTS (d)", tol=0.011))
        ax.axvline(0.5, color=C["line"], ls="--", lw=1.4, zorder=0)
        ax.text(0.5, 1.45, "chance", ha="center", fontsize=11.5, color=C["muted"])
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["NLA", "Oracle Lens"])
        ax.set_ylim(-0.45, 1.55)
        ax.set_xlim(0.2, 1.08)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUROC (0.5 = chance)")
    checks.append(g.check("table column NLA, independent items x independent phrasings vs talk's '0.57'",
                          [c for c in checks if c["quantity"] == "tb_tablecol nla his_x_his"][0]["computed"], 0.57,
                          "talk script S3 hardening ('scored the NLA at 0.57')", tol=0.005))
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.15, -0.2), ncol=3, fontsize=12)
    save(fig, "bk_exp17_cross_scoring",
         title="The independent bank's low table-column score is mostly phrasing, not items",
         caption=("Exp 17 (d): our prompted readers scored on an independent bank of table-column and parenthesis items (84 per row) and "
                  "on ours (160 / 149), with our frozen phrasings (fixed + 3 mined, mean log-odds) or the independent bank's two phrasings. "
                  "AUROC with 95% bootstrap CI clustered by template family (independent items) or passage (ours). On table column the "
                  "NLA goes 0.56 -> 0.85 when only the phrasing changes. Caveat: 84 items per row, wide intervals."),
         section="backup:exp17", role="backup", sources=srcs, checks=checks,
         notes="The independent bank's author is not named per the brief. Independent phrasing example: 'The current excerpt has the first table column.'")


if __name__ == "__main__":
    paren_fourway()
    cross_scoring()
