"""bk_exp20_exhibits_recon (exp 19 exhibits cell re-scored by the reconstructor, post hoc) and
bk_exp21_continuation (text-only continuation predictor of Oracle Lens exhibition)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402

E = style.ROOT / "experiments"
R20 = E / "20_ar_structure" / "results"
NAME = {"py_string": "inside a Python string", "js_string": "inside a JavaScript string", "latex_math": "inside LaTeX math",
        "py_arg": "call argument (1st vs 3rd)", "json_depth": "JSON nesting depth", "html_depth": "HTML nesting depth",
        "md_heading": "Markdown heading level", "md_list": "list nesting depth", "tb_tablecol": "table column"}
ORDER = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "md_heading", "md_list"]


def hanley_mcneil(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return max(0, a - 1.96 * se), min(1, a + 1.96 * se)


def recon(row):
    return json.loads((R20 / f"recon20_{row}.json").read_text())


def fig20():
    s19 = {r["feature"]: r for r in json.loads((E / "19_natural_bank/results/score19.json").read_text())["rows"]}
    fig, ax = plt.subplots(figsize=style.FULL)
    hits_strict = hits_rounded = 0
    for i, row in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        d = recon(row)
        a = d["auroc"]["ol_readout"]
        n = int(d["n"])
        lo, hi = hanley_mcneil(a, n // 2, n // 2)
        pred = s19[row]["cells"]["ol_exhibits"]["predicted_high"]
        judged = s19[row]["cells"]["ol_exhibits"]["observed"]
        ok = pred == (a >= 0.70)
        hits_strict += ok
        hits_rounded += pred == (round(a, 2) >= 0.70)
        col = C["ol"] if ok else C["ink"]
        ax.plot([lo, hi], [y, y], color=col, alpha=0.35, lw=3)
        ax.scatter([a], [y], marker="^" if pred else "v", s=140, facecolor=col if ok else "white", edgecolor=col,
                   lw=2, zorder=3)
        ax.scatter([judged], [y], marker="x", s=70, color=C["muted"], zorder=2)
        ax.text(1.02, y, f"{a:.3f}", va="center", fontsize=12, color=C["ink"])
    ax.axvline(0.70, color=C["muted"], ls="--", lw=1.3)
    ax.axvline(0.5, color=C["line"], ls=":", lw=1.3)
    ax.text(0.70, len(ORDER) - 0.45, "0.70 = counts as high", ha="center", fontsize=11.5, color=C["muted"])
    ax.text(0.5, len(ORDER) - 0.45, "chance", ha="center", fontsize=11, color=C["muted"])
    ax.annotate("0.697: under the line\n(record counts it as 0.70)", (0.697, 4), xytext=(0.30, 3.6), fontsize=12,
                weight="bold", arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.2), va="center")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([NAME[r] for r in reversed(ORDER)])
    ax.set_xlim(-0.03, 1.1)
    ax.set_ylim(-1.1, len(ORDER) - 0.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("probe AUROC on the critic's rebuild of Oracle Lens's free readout (0.5 = chance)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h = [ax.scatter([], [], marker="^", s=110, color=C["ol"], label="predicted high, held"),
         ax.scatter([], [], marker="v", s=110, color=C["ol"], label="predicted low, held"),
         ax.scatter([], [], marker="^", s=110, facecolor="white", edgecolor=C["ink"], lw=2, label="missed"),
         ax.scatter([], [], marker="x", s=70, color=C["muted"], label="pre-registered judged share")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.07, 0.97), fontsize=12, ncol=1)
    checks = [
        {"quantity": "exp 19 exhibits cell via reconstructor, hits at >= 0.70", "computed": f"{hits_strict}/8",
         "record": "7/8", "where": "exp 19 RESULTS post hoc; exp 20 RESULTS; talk S5", "match": hits_strict == 7},
        {"quantity": "same with AUROC rounded to 2 dp before thresholding", "computed": f"{hits_rounded}/8",
         "record": "7/8", "where": "how the record appears to have counted", "match": hits_rounded == 7},
        {"quantity": "call argument Oracle Lens readout AUROC", "computed": round(recon("py_arg")["auroc"]["ol_readout"], 4),
         "record": "0.70 ('all five observed high (0.70 to 0.94)')", "where": "exp 20 RESULTS",
         "match": False},
        {"quantity": "primary scorecard with this cell", "computed": f"{31 - 3 + hits_strict}/38", "record": "35/38",
         "where": "exp 19/20 RESULTS", "match": 31 - 3 + hits_strict == 35},
    ]
    style.save(fig, "bk_exp20_exhibits_recon",
               title="Re-scored through the reconstructor, Oracle Lens's exhibits cell holds on 6 of 8 rows, not 7 (post hoc)",
               caption=("Exp 19's rows: Oracle Lens's free readouts fed to the NLA critic and scored with the frozen probe "
                        "(exp 20), against the 0.70 line used for the other AUROC cells; triangles mark the pre-registered "
                        "direction, crosses the judged share the cell was actually scored with (0.00 on every row). Whiskers are "
                        "Hanley-McNeil 95% intervals (80 vs 80 items; no per-item scores stored, so no bootstrap). Post hoc: the "
                        "measure replaced the judge after it failed. The call-argument row is 0.697, under the line, so the "
                        "strict count is 6/8 (primary 34/38), not the recorded 7/8 (35/38)."),
               section="backup:exp20", role="backup",
               sources=["experiments/20_ar_structure/results/recon20_*.json", "experiments/19_natural_bank/results/score19.json"],
               checks=checks, notes="Threshold convention >= (score19.py: high = v >= CELL_THRESH). New record discrepancy.")
    for c in checks:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


def fig21():
    t = json.loads((E / "21_continuation/results/table21.json").read_text())
    rows = list(t["rows"])
    readable = [r for r in rows if r not in ("md_heading", "md_list")]
    ns = ["1", "2", "4", "8", "16", "32"]
    ol = {r: (recon(r)["auroc"]["ol_readout"]) for r in rows}
    mism = [r for r in rows if abs(ol[r] - t["rows"][r]["exp20"]["ol_readout"]) > 1e-9]
    rho7 = {n: spearmanr([t["rows"][r]["continuation_by_n"][n] for r in readable], [ol[r] for r in readable]).statistic for n in ns}
    rho9 = {n: spearmanr([t["rows"][r]["continuation_by_n"][n] for r in rows], [ol[r] for r in rows]).statistic for n in ns}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=style.FULL, gridspec_kw=dict(width_ratios=[1, 1.25]))
    x = np.arange(len(ns))
    a1.plot(x, [rho9[n] for n in ns], color=C["text_pale"], lw=2, marker="o", ms=8, label="all 9 rows")
    a1.plot(x, [rho7[n] for n in ns], color=C["text"], lw=2.5, marker="o", ms=9, label="7 readable rows")
    a1.axhline(0, color=C["line"], ls="--", lw=1.3)
    a1.set_xticks(x)
    a1.set_xticklabels(ns)
    a1.set_xlabel("continuation tokens used")
    a1.set_ylabel("Spearman with Oracle Lens\nexhibition (exp 20)")
    a1.set_ylim(-0.6, 1.0)
    a1.annotate(f"+{rho7['4']:.2f} at 4 tokens\n(n = 7, not significant)", (2, rho7["4"]), xytext=(2.6, 0.75),
                fontsize=12, arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.1))
    a1.legend(loc="lower left", fontsize=12)
    a1.set_title("rank agreement by length", fontsize=14, loc="left")
    for r in rows:
        c4 = t["rows"][r]["continuation_by_n"]["4"]
        key = r in ("py_string", "tb_tablecol")
        a2.scatter([c4], [ol[r]], s=110 if key else 70, color=C["text"] if key else C["text_pale"],
                   edgecolor=C["ink"] if key else "none", zorder=3)
        off = {"py_string": (-0.02, 0.02, "right"), "tb_tablecol": (0.0, -0.04, "center"),
               "md_heading": (0.015, -0.02, "left"), "md_list": (0.015, 0.0, "left"),
               "html_depth": (-0.01, -0.045, "center"), "json_depth": (0.0, 0.017, "center"),
               "py_arg": (0.012, -0.02, "left"), "latex_math": (0.012, -0.025, "left"),
               "js_string": (-0.015, -0.02, "right")}[r]
        a2.text(c4 + off[0], ol[r] + off[1], NAME[r].replace(" (1st vs 3rd)", "") + (" (synthetic)" if r == "tb_tablecol" else ""),
                ha=off[2], fontsize=11.5 if not key else 12.5, weight="bold" if key else "normal")
    a2.plot([0.5, 1], [0.5, 1], color=C["line"], ls=":", lw=1.2)
    a2.set_xlim(0.5, 1.02)
    a2.set_ylim(0.5, 1.0)
    a2.set_xlabel("next 4 tokens predict the state (AUROC)")
    a2.set_ylabel("Oracle Lens exhibits it (AUROC)")
    a2.set_title("both separable in the continuation; lens 0.91 vs 0.64", fontsize=14, loc="left")
    fig.subplots_adjust(wspace=0.38)
    checks = [{"quantity": "exp 20 Oracle Lens readout copied into table21 equals recon20", "computed": f"{len(mism)} mismatches",
               "record": "0", "where": "table21.json vs recon20_*.json", "match": not mism},
              {"quantity": "Spearman readable 7 at 4 tokens", "computed": round(float(rho7["4"]), 3), "record": "+0.54",
               "where": "exp 21 RESULTS", "match": bool(round(rho7["4"], 2) == 0.54)},
              {"quantity": "Spearman all 9 at 4 tokens", "computed": round(float(rho9["4"]), 3), "record": "+0.78",
               "where": "exp 21 RESULTS", "match": bool(round(rho9["4"], 2) == 0.78)},
              {"quantity": "Spearman readable 7 at other lengths", "computed": {n: round(float(rho7[n]), 2) for n in ns if n != "4"},
               "record": "near zero or negative", "where": "exp 21 RESULTS",
               "match": bool(all(rho7[n] < 0.1 for n in ns if n != "4"))},
              {"quantity": "string state vs table column Oracle Lens readout", "computed": f"{ol['py_string']:.2f} vs {ol['tb_tablecol']:.2f}",
               "record": "0.91 vs 0.64", "where": "exp 21 RESULTS", "match": f"{ol['py_string']:.2f} vs {ol['tb_tablecol']:.2f}" == "0.91 vs 0.64"}]
    style.save(fig, "bk_exp21_continuation",
               title="Whether the continuation changes does not predict what Oracle Lens exhibits",
               caption=("Exp 21 (exploratory, post hoc): a bag-of-tokens classifier on the true next n tokens of the source, "
                        "grouped 5-fold by file. Left: Spearman between that AUROC and Oracle Lens's exhibition (exp 20's "
                        "reconstructor AUROC) across rows; only 4 tokens over the 7 readable rows is positive (+0.54, n = 7, not "
                        "significant), the 9-row +0.78 is carried by the two unreadable Markdown rows. Right: at 4 tokens every "
                        "readable row is separable (0.85 to 0.97), yet the lens exhibits the Python string at 0.91 and the table "
                        "column at 0.64. Table column's continuation is synthetic (regenerated from exp 13's cell pools). No CIs: "
                        "per-row AUROCs only, n = 7 to 9 rows."),
               section="backup:exp21", role="backup",
               sources=["experiments/21_continuation/results/table21.json", "experiments/20_ar_structure/results/recon20_*.json"],
               checks=checks, notes="Spearman recomputed with scipy from per-row values.")
    for c in checks:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    fig20()
    fig21()
