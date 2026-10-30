"""s6_textbag_check (exp 20): a bag-of-tokens classifier on the readout text alone (no critic) agrees with the critic
on which readouts carry which states. Text-bag AUROC recomputed from the raw readouts with out-of-fold scores."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402
from g4_s6_carry_ladder import MARKED, ROWS  # noqa: E402

E20 = G.EXP / "20_ar_structure" / "results"
TALK_OL = {"py_string": 0.96, "js_string": 0.88, "latex_math": 0.78, "py_arg": 0.72, "json_depth": 0.94,
           "html_depth": 0.84, "tb_tablecol": 0.71, "tb_paren": 0.56}
SHORT = dict(G.SHORT, json_depth="JSON depth", html_depth="HTML depth", tb_paren="parentheses")


def compute():
    sources, checks, data = [], [], {}
    rec_tb = G.load(E20 / "readout_textbag.json")
    sources.append(G.rel(E20 / "readout_textbag.json"))
    for row in ROWS:
        base = G.EXP / ("13_new_rows" if row.startswith("tb_") else "19_natural_bank") / "results" / "items_base.json"
        items = [it for it in G.load(base) if it["feature"] == row and it["split"] == "select"]
        sources.append(G.rel(base))
        y = np.array([it["value"] == MARKED[row] for it in items], int)
        g = np.array([it["cluster"] for it in items])
        crit = G.load(E20 / f"recon20_{row}.json")["auroc"]
        data[row] = {}
        for reader in ("ol", "nla"):
            f = E20 / f"{reader}_sel_{row}.json"
            sources.append(G.rel(f))
            texts = G.load(f)[reader]
            s = G.text_bag_scores([texts[it["id"]] for it in items], y, g)
            a, lo, hi = G.auroc_ci(s, y, g)
            data[row][reader] = dict(bag=a, lo=lo, hi=hi, critic=crit[f"{reader}_readout"])
            checks.append(G.check(f"{row} {G.READER[reader]} text bag", a, round(rec_tb[row][f"{reader}_text_bag"], 4), "exp20 results/readout_textbag.json", tol=0.001))
            if reader == "ol" and row in TALK_OL:
                checks.append(G.check(f"{row} Oracle Lens text bag (2dp)", round(a, 2), TALK_OL[row], "talk script S6 decorrelated check"))
        print(row, {k: round(v["bag"], 3) for k, v in data[row].items()}, flush=True)
    return data, sorted(set(sources)), checks


def main():
    data, sources, checks = compute()
    style.slide_style()
    fig, axes = plt.subplots(1, 2, figsize=G.FIG, sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    for ax, reader, col, pale, mk, name in ((axes[0], "ol", C["ol"], C["ol_pale"], "D", "Oracle Lens readout"),
                                            (axes[1], "nla", C["nla"], C["nla_pale"], "o", "NLA explanation")):
        ax.plot([0.4, 1.02], [0.4, 1.02], color=C["line"], lw=1.3, zorder=0)
        ax.axhline(0.5, color=C["line"], lw=1.2, ls="--", zorder=0)
        ax.axvline(0.5, color=C["line"], lw=1.2, ls="--", zorder=0)
        for row in ROWS:
            d = data[row][reader]
            md = row in G.MARKDOWN
            ax.plot([d["critic"]] * 2, [d["lo"], d["hi"]], color=pale, lw=2, zorder=1)
            ax.plot(d["critic"], d["bag"], mk, ms=8, mfc="white" if md else col, mec=pale if md else col, mew=2, zorder=3)
        ax.set_title(name, color=col)
        ax.set_xlim(0.4, 1.03)
        ax.set_ylim(0.4, 1.03)
        ax.set_xlabel("through the critic (AUROC)")
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    # selected labels
    lab = {"ol": {"py_string": (-10, 6, "right"), "tb_tablecol": (-8, 8, "right"),
                  "tb_paren": (8, -6, "left"), "py_arg": (8, -12, "left")},
           "nla": {"py_arg": (-8, 0, "right"), "tb_paren": (-8, 0, "right")}}
    for ax, reader in zip(axes, ("ol", "nla")):
        for row, (dx, dy, ha) in lab[reader].items():
            d = data[row][reader]
            ax.annotate(SHORT[row], (d["critic"], d["bag"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=15, color=C["muted"] if row in G.MARKDOWN else C["ink"])
    axes[0].set_ylabel("from the words alone\n(word-count classifier, AUROC)")
    axes[1].text(0.42, 1.01, "hollow = Markdown\n(not linearly present)", ha="left", va="top", fontsize=15, color=C["muted"],
                 linespacing=1.05)
    axes[0].text(1.0, 0.42, "diagonal:\nthe two agree", ha="right", va="bottom", fontsize=15, color=C["muted"],
                 linespacing=1.05)
    axes[1].text(1.0, 0.42, "critic > words on\ncall argument", ha="right", va="bottom", fontsize=15, color=C["nla"],
                 linespacing=1.05)
    style.save(
        fig, "s6_textbag_check",
        title="A plain word-count classifier with no critic finds the same states in the readouts",
        caption=("Per row of exp 20 (160 selection items, k = 1 readouts), x = frozen-probe AUROC of the critic's rebuild from "
                 "the readout, y = AUROC of a binary bag-of-tokens logistic classifier on the readout text alone (grouped "
                 "5-fold by source file, recomputed here from the saved readouts). Bars: cluster-bootstrap 95% over source "
                 "files (y only; the critic has no per-item scores). Oracle Lens sits on the diagonal; the NLA's bag is "
                 "lower than its in-distribution critic on a few rows (call argument 0.97 by the critic vs 0.67 by the words: the critic reads more than the words, or it knows NLA prose). Hollow points are the two Markdown rows (not linearly present at layer 42)."),
        section="S6", role="main", sources=sources, checks=checks,
        notes=("Classifier: lib19.bag_of_tokens (window all, min_df 3) on Qwen3.6-27B tokens, mats.probes logistic lam 1.0, "
               "group_folds seed 0, exactly exp 23's carrying classifier; the recomputation reproduces readout_textbag.json."),
    )
    print("mismatch", [c for c in checks if not c["match"]])


if __name__ == "__main__":
    main()
