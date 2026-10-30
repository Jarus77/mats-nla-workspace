"""s5_what_failed_in_18: exp 18, how much the model uses a state (|z|) vs how often the NLA states it."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402

R = style.ROOT / "experiments" / "18_prediction_test" / "results"
NAMES = {"json_depth": "JSON nesting depth", "latex_math": "LaTeX math", "py_string": "Python string",
         "call_arg": "call argument", "md_heading": "Markdown heading", "csv_col": "CSV column",
         "html_depth": "HTML nesting depth", "stance_word": "stance word", "ij2_lang": "natural language",
         "bk_quote": "inside a quote"}
NA_X = -4.6  # strip for rows without a next-token set (used z not defined)
NA_JIT = {"md_heading": -1.8, "stance_word": 1.8, "ij2_lang": 0.0}
NA_HALF = 3.4


def stated(feature):
    j = json.loads((R / "judged" / f"{feature}_nla.json").read_text())
    v = np.array([bool(x["stated"]) for x in j["judged"]], float)
    items = np.array([x["item"] for x in j["judged"]])
    lo, hi = style.bootstrap_ci(np.mean, v, groups=items)
    return float(v.mean()), lo, hi, len(v), len(set(items))


def main():
    instr = json.loads((R / "instruments18.json").read_text())
    score = {r["feature"]: r for r in json.loads((R / "score18.json").read_text())["rows"]}
    pts = []
    for row in instr:
        f = row["feature"]
        z = row["instr"]["used_z"]
        p, lo, hi, n, ni = stated(f)
        pts.append(dict(f=f, z=None if z is None or np.isnan(z) else abs(z), zs=z, p=p, lo=lo, hi=hi, n=n, ni=ni,
                        pred=score[f]["cells"]["nla_states"]["predicted_high"],
                        rec=score[f]["cells"]["nla_states"]["observed"]))
    style.slide_style()
    fig, ax = plt.subplots(figsize=(style.SLIDE[0] + 0.4, style.SLIDE[1] - 0.1))
    ax.axvspan(NA_X - NA_HALF, NA_X + NA_HALF, color=C["line"], alpha=0.3, zorder=0)
    ax.text(NA_X, 1.04, "no next-token\ntest", ha="center", va="bottom", fontsize=15, color=C["muted"])
    ax.axvline(3, color=C["muted"], ls="--", lw=1.3, zorder=0)
    ax.text(3.4, 1.06, "|z| = 3: 'used'", fontsize=15, color=C["muted"], va="bottom")
    ax.axhline(0.20, color=C["muted"], ls="--", lw=1.3, zorder=0)
    ax.text(21.0, 0.19, "0.20 = counts as stated", fontsize=15, color=C["muted"], ha="right", va="top")
    for q in pts:
        x = NA_X + NA_JIT[q["f"]] if q["z"] is None else q["z"]
        correct = q["pred"] == (q["p"] >= 0.20)
        col = C["nla"] if correct else C["ink"]
        ax.plot([x, x], [q["lo"], q["hi"]], color=col, alpha=0.4, lw=2, zorder=1)
        ax.scatter([x], [q["p"]], s=70, color=col if correct else "white", edgecolor=col, lw=2, zorder=3)
    lab = {"py_string": (0.7, 0.0, "left"), "call_arg": (0.8, 0.04, "left"), "json_depth": (0.5, 0.22, "left"),
           "bk_quote": (1.0, 0.0, "left"), "csv_col": (-1.0, 0.27, "right")}
    for f, txt, dx, y, ha in [("md_heading", "heading", 0, -0.085, "center"), ("stance_word", "stance", 0, 0.085, "center"),
                              ("html_depth", "HTML nesting depth", 2.6, -0.085, "left")]:
        q = next(x for x in pts if x["f"] == f)
        x = NA_X + NA_JIT[f] if q["z"] is None else q["z"]
        if dx:
            ax.annotate(txt, (x, 0.0), xytext=(x + dx, y), ha=ha, va="center", fontsize=15, color=C["ink"],
                        arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8, shrinkB=6))
        else:
            ax.text(x, y, txt, ha=ha, va="center", fontsize=15, color=C["ink"])
    for q in pts:
        if q["f"] in lab:
            dx, dy, ha = lab[q["f"]]
            x = NA_X + NA_JIT[q["f"]] if q["z"] is None else q["z"]
            ax.annotate(NAMES[q["f"]], (x, q["p"]), xytext=(x + dx, q["p"] + dy), ha=ha, va="center", fontsize=15,
                        arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    ax.annotate("LaTeX math:\nmost used,\nnever stated", (22.0, 0.03), xytext=(22.0, 0.50), fontsize=15, weight="bold", va="center",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.3))
    ax.annotate("language: not used,\nstated 95%", (NA_X + 0.5, 0.95), xytext=(4.0, 0.86), fontsize=15, weight="bold",
                va="center",
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.3))
    h1 = ax.scatter([], [], s=70, color=C["nla"], label="rule right")
    h2 = ax.scatter([], [], s=70, color="white", edgecolor=C["ink"], lw=2, label="rule wrong")
    ax.legend(handles=[h1, h2], loc="upper right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
    ax.set_xlim(NA_X - NA_HALF, 26)
    ax.set_ylim(-0.14, 1.17)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("how much the model uses the state (|z| of the next-token push)")
    ax.set_ylabel("NLA states it\n(share of 320 readouts)")

    checks = []
    for q in pts:
        checks.append({"quantity": f"{q['f']} stated rate from judged verdicts", "computed": round(q["p"], 4),
                       "record": round(q["rec"], 4), "where": "score18.json nla_states", "match": abs(q["p"] - q["rec"]) < 1e-6})
    d = {q["f"]: q for q in pts}
    checks += [
        {"quantity": "LaTeX math used z, stated", "computed": f"{d['latex_math']['zs']:.2f}, {d['latex_math']['p']:.2f}",
         "record": "+22.15, 0.00 ('most used row, never stated')", "where": "predictions18.md; talk S5",
         "match": round(d['latex_math']['zs'], 2) == 22.15 and d['latex_math']['p'] == 0},
        {"quantity": "language stated rate", "computed": round(d["ij2_lang"]["p"], 3), "record": "0.95", "where": "talk S5",
         "match": round(d["ij2_lang"]["p"], 2) == 0.95},
        {"quantity": "rule 'NLA states exactly when used' hits", "computed": f"{sum(q['pred'] == (q['p'] >= 0.2) for q in pts)}/10",
         "record": "5/10", "where": "talk S5 table", "match": sum(q['pred'] == (q['p'] >= 0.2) for q in pts) == 5},
    ]
    title = "Exp 18: the NLA does not state what the model uses"
    caption = ("Exp 18's ten synthetic rows: x = how strongly pushing the state moves the model's next-token odds (|z| vs 20 "
               "isotropic nulls; grey strip = rows with no next-token set), y = share of the NLA's 320 free readouts (40 items "
               "x 8) a judge marked as stating the property, with item-clustered bootstrap 95% intervals. The pre-registered rule "
               "'states exactly when used' (it predicts stated right of the |z| = 3 line) got 5 of 10: LaTeX math is the most used row and is never "
               "stated, natural language has no next-token use and is stated 95%. 'Used' is the language here; the NLA's "
               "statements followed the statement shift instead (exp 28 later showed that only separates language from structure).")
    notes = ("Computed from instruments18.json (used_z, two-sided rule |z| >= 3 per prereg §2 clarification), judged/<row>_nla.json "
             "(per-readout 'stated' verdicts, n = 320, 40 items), score18.json (predicted cell). Stated threshold 0.20 from "
             "notes/prereg_prediction_test.md. Python string has z = -12.2 (used, negative direction).")
    style.save(fig, "s5_what_failed_in_18", title=title, caption=caption, section="S5", role="main",
               sources=["experiments/18_prediction_test/results/instruments18.json",
                        "experiments/18_prediction_test/results/judged/*_nla.json",
                        "experiments/18_prediction_test/results/score18.json"], checks=checks, notes=notes)
    for c in checks:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    main()
