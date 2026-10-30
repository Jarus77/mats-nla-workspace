"""s3_sixway (exp 17a): six-candidate re-score of table column. NLA argmax per true class, class AUROC per reader."""
from __future__ import annotations

import g2_common as g
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp
import style
from style import C, auroc, bootstrap_ci, save

VOL = g.EXP / "17_harden" / "results" / "vol"
CLASS = {"first": [0], "late": [3, 4, 5]}  # spec17 CLASS_SETS


def lse_contrast(lp):
    return logsumexp(lp[:, CLASS["late"]], 1) - logsumexp(lp[:, CLASS["first"]], 1)


def class_auroc(lp, y_first):
    c = lse_contrast(lp)
    return auroc(c, ~y_first)  # higher contrast = late


def main() -> None:
    nla = torch.load(VOL / "mc_nla_tb_tablecol.pt", weights_only=False, map_location="cpu")
    ol = torch.load(VOL / "mc_ol_tb_tablecol.pt", weights_only=False, map_location="cpu")
    y = np.asarray(nla["y"]).astype(int)  # 0 = first (values[0]), 1 = late; check below
    truth = np.asarray(nla["truth"])
    cl = np.asarray(nla["clusters"])
    lp = nla["lp"]["nla:mc:fixed"].numpy().astype(float)
    am = lp.argmax(1)
    first_mask = truth == 0
    assert np.array_equal(first_mask, y == 0), "y/truth convention"
    cnt_first = np.bincount(am[first_mask], minlength=6)
    cnt_late = np.bincount(am[~first_mask], minlength=6)
    top1 = (am == truth).astype(float)
    t1 = top1.mean()
    t1ci = bootstrap_ci(lambda a: float(a.mean()), top1, groups=cl)

    cells = [("NLA", "nla", lp), ("Oracle Lens", "ol", ol["lp"]["ol:mc:m3"].numpy().astype(float)),
             ("J-lens", "jl", ol["lp"]["jlens:mc:fixed"].numpy().astype(float))]
    yo = np.asarray(ol["y"]).astype(int)
    res = []
    for lab, ck, L in cells:
        yy = (y if ck == "nla" else yo) == 0
        clu = cl if ck == "nla" else np.asarray(ol["clusters"])
        a = class_auroc(L, yy)
        ci = bootstrap_ci(lambda Lb, yb: class_auroc(Lb, yb), L, yy, groups=clu)
        res.append((lab, ck, a, ci))
    ol_all = {k: class_auroc(v.numpy().astype(float), yo == 0) for k, v in ol["lp"].items()}

    style.slide_style()
    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.42, 0.95], wspace=0.12, left=0.075, right=0.985, bottom=0.25, top=0.88)
    names = ["1", "2", "3", "4", "5", "6"]
    x = np.arange(6)
    for col, (cnt, title, n) in enumerate(((cnt_first, "A. first-cell items", int(first_mask.sum())),
                                           (cnt_late, "B. late-cell items", int((~first_mask).sum())))):
        ax = fig.add_subplot(gs[0, col])
        cols = [C["nla"] if (col == 0 and i == 0) else (C["nla_pale"] if cnt[i] else C["line"]) for i in range(6)]
        if col == 1:
            cols = [C["nla_pale"] if i < 3 else C["line"] for i in range(6)]
        bars = ax.bar(x, cnt, color=cols, width=0.72)
        if col == 1:
            ax.axvspan(2.5, 5.5, color=C["line"], alpha=0.3, zorder=0)
            ax.text(4.0, 72, "true\ncolumns,\nnever\nchosen", ha="center", va="center", fontsize=15,
                    color=C["muted"], linespacing=1.05)
        for b, v in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}", ha="center", va="bottom", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(0, 92)
        ax.set_yticks([0, 40, 80])
        ax.set_title(title, loc="left", x=-0.12 if col == 0 else 0.0)
        if col == 0:
            ax.set_ylabel("items")
            ax.text(0.7, 62, f"'first' on\n{100 * cnt[0] / n:.0f}%", fontsize=15, color=C["nla"], ha="left",
                    va="center", linespacing=1.05)
        else:
            ax.set_yticklabels([])
    fig.text(0.33, 0.1, "column the NLA's phrasing ranks highest (of six)", ha="center", va="center", fontsize=15)

    bx = fig.add_subplot(gs[0, 3])
    yp = np.arange(len(res))[::-1]
    for (lab, ck, a, ci), yy in zip(res, yp):
        bx.plot(ci, [yy, yy], color=C[ck], lw=3, alpha=0.55)
        bx.plot(a, yy, "o", ms=11, color=C[ck], mec="white", mew=1.2)
        bx.text(a, yy + 0.2, f"{a:.2f}", ha="center", va="bottom", fontsize=15, color=C[ck])
    bx.axvline(0.5, color=C["line"], ls="--", lw=1.5, zorder=0)
    bx.axvline(0.8, color=C["line"], ls=":", lw=1.2, zorder=0)
    bx.text(0.5, -0.75, "chance", ha="center", fontsize=15, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
    bx.text(0.8, -0.75, "0.80", ha="center", fontsize=15, color=C["muted"])
    bx.set_yticks(yp)
    bx.set_yticklabels([r[0] for r in res])
    bx.set_xlim(0.4, 1.0)
    bx.set_xticks([0.4, 0.6, 0.8, 1.0])
    bx.set_ylim(-0.85, len(res) - 0.35)
    bx.set_xlabel("first vs late AUROC,\nall six candidates")
    bx.set_title("C. Six candidates", loc="left", x=-0.5)

    checks = [
        g.check("NLA 'first' argmax on first-cell items", f"{cnt_first[0]}/{first_mask.sum()}", "75/80", "exp 17 RESULTS (a)"),
        g.check("NLA share 'first' on first-cell items", round(cnt_first[0] / first_mask.sum(), 2), 0.94, "talk script S3 hardening"),
        g.check("NLA argmax hits on columns 4-6 (all items)", int(cnt_first[3:].sum() + cnt_late[3:].sum()), 0, "talk script 'never names columns four to six'"),
        g.check("NLA late items argmax first/second/third", cnt_late[:3].tolist(), [39, 39, 2], "exp 17 mc17.md"),
        g.check("NLA six-way class AUROC", round(res[0][2], 2), 0.89, "talk script S3 / exp 17 RESULTS"),
        g.check("NLA six-way class AUROC CI", [round(v, 2) for v in res[0][3]], [0.84, 0.94], "exp 17 RESULTS (a)"),
        g.check("NLA six-way top-1", round(t1, 2), 0.47, "exp 17 RESULTS (a)"),
        g.check("Oracle Lens col_N class AUROC", round(res[1][2], 2), 0.80, "exp 17 RESULTS (a)"),
        g.check("J-lens class AUROC", round(res[2][2], 2), 0.66, "exp 17 RESULTS (a)"),
        {"quantity": "source of the 94%", "computed": "prompted six-way argmax of 'The current cell is in the {v} column of the table'",
         "record": "talk script S3 cross-question 1: 'the NLA's free readouts name \"first\" unprompted on 94% of first-cell items'",
         "where": "talk_script.md S3 cross-questions", "match": False},
    ]
    save(fig, "s3_sixway",
         title="With six columns to choose from, the NLA still separates first from late but never picks columns 4 to 6",
         caption=(f"Exp 17: table-column items re-scored over all six column values (160 exp 13 selection items: 80 first-cell, 80 late-cell with true column 4-6). A and B: which "
                  f"column the NLA's phrasing 'The current cell is in the {{v}} column of the table' ranks highest; it picks 'first' "
                  f"on {cnt_first[0]} of 80 first-cell items and never picks columns 4-6, even when they are true. C: first vs "
                  "late AUROC from log-sum-exp over each class's candidates (Oracle Lens: column-number bullets; J-lens: column words), 95% bootstrap CI clustered by passage (2,000 resamples). "
                  f"Six-way top-1 for the NLA is {t1:.2f} [{t1ci[0]:.2f}, {t1ci[1]:.2f}] against chance 0.17. Caveat: the 94% is a "
                  "prompted argmax, not free readouts; with a per-candidate prior subtracted, late items do go to columns 4-6 (45 of 80)."),
         section="S3", role="main",
         sources=[g.rel(VOL / "mc_nla_tb_tablecol.pt"), g.rel(VOL / "mc_ol_tb_tablecol.pt"),
                  "experiments/17_harden/spec17.py (CLASS_SETS, templates)"],
         checks=checks,
         notes=(f"Oracle Lens other phrasings: class AUROC {', '.join(f'{k} {v:.2f}' for k, v in ol_all.items())}; the panel shows "
                "the col_N phrasing (m3) that RESULTS quotes. Parenthesis four-way re-score failed the declared bar (top-1 lower bound "
                "0.33 < 0.50): see bk_exp17_paren_fourway. The talk script's cross-question 1 attributes the 94% to free readouts; it is "
                "from the prompted six-way argmax (free-readout stated rate on table column is 41%, exp 13)."))


if __name__ == "__main__":
    main()
