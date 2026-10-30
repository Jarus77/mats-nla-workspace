"""bk_exp18_rows / bk_exp19_rows / bk_exp22_rows: per-row observed value vs threshold, predicted H/L, hit or miss.
Aggregates only (no item text)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402

E = style.ROOT / "experiments"
DIRS = {"18": "18_prediction_test", "19": "19_natural_bank", "22": "22_public_bank"}
THRESH = {"nla_prompted": 0.85, "nla_states": 0.20, "nla_names_complement": 0.20, "ol_prompted": 0.70,
          "ol_exhibits": 0.30, "ol_states": 0.05, "jlens_prompted": 0.70}  # feature_spec18/19/22.CELL_THRESH
CELLS = list(THRESH)
CELL_LABEL = {"nla_prompted": "NLA prompted\n(AUROC)", "nla_states": "NLA states\n(share)",
              "nla_names_complement": "NLA complement\n(multi-choice)", "ol_prompted": "Oracle Lens\nprompted (AUROC)",
              "ol_exhibits": "Oracle Lens exhibits\n(judged share)", "ol_states": "Oracle Lens\nstates (share)",
              "jlens_prompted": "J-lens prompted\n(AUROC)"}
READER = {"nla": "nla", "ol_": "ol", "jle": "jl"}
ROWNAME = {"json_depth": "JSON nesting depth", "latex_math": "inside LaTeX math", "py_string": "inside a Python string",
           "call_arg": "call argument", "md_heading": "Markdown heading level", "csv_col": "CSV column",
           "html_depth": "HTML nesting depth", "stance_word": "stance word", "ij2_lang": "natural language",
           "bk_quote": "inside a quote", "js_string": "inside a JavaScript string", "py_arg": "call argument",
           "md_list": "list nesting depth", "code_lang": "Python vs JavaScript (control)"}
CUED = {"19": {"json_depth", "html_depth"}, "22": {"json_depth", "code_lang"}, "18": set()}


def judged_ci(test, row, reader, kind):
    p = E / DIRS[test] / "results" / "judged" / f"{row}_{reader}.json"
    d = json.loads(p.read_text())
    j, vals = d["judged"], d["values"]
    if kind == "stated":
        v = np.array([bool(x["stated"]) for x in j], float)
    else:
        code = {vals[0]: "A", vals[1]: "B"}  # judge codes the exhibited value as A (first) / B (second)
        v = np.array([x["exhibited_value"] == code[x["value"]] for x in j], float)
    items = np.array([x["item"] for x in j])
    lo, hi = style.bootstrap_ci(np.mean, v, groups=items)
    return float(v.mean()), lo, hi


def make(test):
    s = json.loads((E / DIRS[test] / "results" / f"score{test}.json").read_text())
    rows = s["rows"]
    fig, axes = plt.subplots(1, 7, figsize=(15, 0.62 * len(rows) + 2.4), sharey=True)
    ys = {r["feature"]: len(rows) - 1 - i for i, r in enumerate(rows)}
    checks, n_ok = [], 0
    for ax, cell in zip(axes, CELLS):
        rk = READER[cell[:3]]
        for r in rows:
            c = r["cells"][cell]
            y = ys[r["feature"]]
            v = c["observed"]
            ci = None
            if cell in ("nla_states", "ol_states", "ol_exhibits"):
                reader = "nla" if cell == "nla_states" else "ol42"
                m, lo, hi = judged_ci(test, r["feature"], reader, "exh" if cell == "ol_exhibits" else "stated")
                ci = (lo, hi)
                ok = abs(m - v) < 1e-6
                n_ok += ok
                if not ok:
                    checks.append({"quantity": f"exp {test} {r['feature']} {cell} from judged", "computed": m,
                                   "record": v, "where": f"score{test}.json", "match": False})
            assert (v >= THRESH[cell]) == c["observed_high"]
            col = C[rk] if c["correct"] else C["ink"]
            if ci:
                ax.plot(ci, [y, y], color=col, alpha=0.35, lw=2.5, zorder=1)
            marker = "^" if c["predicted_high"] else "v"
            scored = not (cell == "jlens_prompted" and r["feature"] in CUED[test])
            face = col if c["correct"] else "white"
            if not scored:  # lexically cued row: J-lens cell not scored in the primary scorecard
                face, col = C["line"], C["muted"]
            ax.scatter([v], [y], marker=marker, s=110, facecolor=face, edgecolor=col, lw=1.8, zorder=3)
        ax.axvline(THRESH[cell], color=C["muted"], ls="--", lw=1.2, zorder=0)
        hits = sum(r["cells"][cell]["correct"] for r in rows)
        ax.set_title(f"{CELL_LABEL[cell]}\n{hits}/{len(rows)} right", fontsize=12.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "", "1"], fontsize=11)
        ax.text(THRESH[cell], len(rows) - 0.45, f"{THRESH[cell]:.2f}".rstrip("0").rstrip("."), ha="center",
                va="bottom", fontsize=10.5, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
        ax.tick_params(axis="y", length=0)
        if ax is not axes[0]:
            ax.spines["left"].set_visible(False)
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([ROWNAME[f] for f in ys], fontsize=12.5)
    axes[0].set_ylim(-0.6, len(rows) - 0.1)
    h = [axes[0].scatter([], [], marker="^", s=100, color=C["muted"], label="predicted high"),
         axes[0].scatter([], [], marker="v", s=100, color=C["muted"], label="predicted low"),
         axes[0].scatter([], [], marker="o", s=100, facecolor="white", edgecolor=C["ink"], lw=1.8,
                         label="open = prediction missed")]
    if CUED[test]:
        h.append(axes[0].scatter([], [], marker="o", s=100, facecolor=C["line"], edgecolor=C["muted"], lw=1.8,
                                 label="grey = cued row, J-lens not scored"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.04),
               fontsize=12.5)
    fig.subplots_adjust(wspace=0.15)
    acc = s["accuracy"]
    checks.insert(0, {"quantity": f"exp {test} judged-cell observed values recomputed from per-readout verdicts",
                      "computed": f"{n_ok}/{3 * len(rows)} match", "record": f"{3 * len(rows)}/{3 * len(rows)}",
                      "where": f"score{test}.json", "match": n_ok == 3 * len(rows)})
    checks.append({"quantity": f"exp {test} all cells", "computed": f"{acc['correct']}/{acc['n']}",
                   "record": {"18": "44/70", "19": "44/56", "22": "44/56"}[test], "where": "RESULTS.md",
                   "match": f"{acc['correct']}/{acc['n']}" == {"18": "44/70", "19": "44/56", "22": "44/56"}[test]})
    desc = {"18": "Exp 18 (10 synthetic rows)", "19": "Exp 19 (8 natural-text rows, unpublished bank)",
            "22": "Exp 22 (public rebuild, 8 rows incl. Python-vs-JavaScript control)"}[test]
    title = {"18": "Exp 18 missed its bar on the J-lens, NLA complement and NLA states cells (44 of 70)",
             "19": "Exp 19: misses sit in Oracle Lens exhibits and the prompted cells (44 of 56)",
             "22": "Exp 22: the control row is the one the NLA states, Oracle Lens exhibits and the J-lens reads"}[test]
    caption = (f"{desc}: each panel is one pre-registered cell; markers show the observed value per row, triangle up/down "
               "= predicted high/low, dashed line = the declared 'counts as high' threshold; filled in the reader's colour "
               "when the prediction held, open dark when it missed. Horizontal bars on the three judged cells are "
               "item-clustered bootstrap 95% intervals (320 readouts, 40 items); the AUROC cells have no per-item scores "
               "stored, so no interval. 'NLA complement (multi-choice)' is the pre-registered cell 'NLA names the complement': a multi-candidate forced-choice argmax on the unmarked value, not a regex or judge verdict. Grey J-lens markers are lexically cued rows, excluded from the primary scorecard as declared (still counted in the all-cells total).")
    style.save(fig, f"bk_exp{test}_rows", title=title, caption=caption, section=f"backup:exp{test}", role="backup",
               sources=[f"experiments/{DIRS[test]}/results/score{test}.json",
                        f"experiments/{DIRS[test]}/results/judged/*.json"], checks=checks,
               notes="Observed-high verified as observed >= feature_spec CELL_THRESH for every cell. Exp 19 items never read "
                     "beyond verdict flags; no item text plotted.")
    for c in checks:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    for t in ["18", "19", "22"]:
        make(t)
