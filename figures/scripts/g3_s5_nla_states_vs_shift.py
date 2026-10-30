"""s5_nla_states_vs_shift: exp 28's 47 rows, paraphrase-mean statement shift vs NLA stated rate."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import C, plt  # noqa: E402

D = style.ROOT / "experiments" / "28_stmt_curve" / "results"
NAMES = {"tb_tablecol": "table column", "tb_paren": "parenthesis", "call_arg": "call argument",
         "bk_listidx": "list item index", "ij2_lang": "language", "code_lang": "Python vs JavaScript",
         "bk_code": "code language", "nj2_tense": "tense", "bk_caps": "capitals"}


def perm_p(x, y, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def main():
    meta = {r["key"]: r for r in json.loads((D / "rows28.json").read_text())}
    rows = []
    for f in sorted(D.glob("stmt28_*.json")):
        s = json.loads(f.read_text())
        m = meta[s["key"]]
        sh = s["shift"]
        k = round(s["stated"] * m["n_readouts"])
        rows.append(dict(key=s["key"], feat=s["feature"], exp=s["experiment"], type=s["type"], stated=s["stated"],
                         n=m["n_readouts"], k=k, mean=sh["para_mean"], lo=sh["para_min"], hi=sh["para_max"],
                         rng=sh["para_range"], fixed=sh["fixed"], counter=m["counter_row"], lang=m["language_row"]))
    n_rows = len(rows)
    st = [r for r in rows if r["type"] == "structural"]
    si = [r for r in rows if r["type"] != "structural"]
    rho_all, p_all = perm_p([r["mean"] for r in rows], [r["stated"] for r in rows])
    rho_st, p_st = perm_p([r["mean"] for r in st], [r["stated"] for r in st])
    rng15 = sorted([r for r in rows if r["rng"] > 0.15], key=lambda r: -r["rng"])
    max_struct_shift = max(r["mean"] for r in st)

    style.slide_style()
    fig, ax = plt.subplots(figsize=(style.SLIDE[0] + 0.3, style.SLIDE[1]))
    for group, col, face in [(st, C["nla"], C["nla"]), (si, C["probe"], "white")]:
        for r in group:
            p, lo, hi = style.wilson(r["k"], r["n"])
            ax.plot([r["lo"], r["hi"]], [r["stated"]] * 2, color=col, alpha=0.22, lw=1.3, zorder=1)
            ax.plot([r["mean"]] * 2, [lo, hi], color=col, alpha=0.22, lw=1.3, zorder=1)
        ax.scatter([r["mean"] for r in group], [r["stated"] for r in group], s=46, facecolor=face,
                   edgecolor=col, lw=1.6, zorder=3)
    ax.axvline(0.30, color=C["muted"], ls="--", lw=1.4, zorder=0)
    ax.text(0.305, -0.02, "0.30 rule (withdrawn)", fontsize=15, color=C["muted"], va="bottom")
    ax.axvspan(-0.07, max_struct_shift, color=C["nla_pale"], alpha=0.18, zorder=0)
    ax.text(-0.065, 0.74, f"{len(st)} structural rows,\nevery shift ≤ {max_struct_shift:.2f}", fontsize=15,
            color=C["nla"], weight="bold", va="center", ha="left")
    offs = {"13_tb_tablecol": (0.075, 0.45, "left"), "18_call_arg": (0.10, 0.355, "left"),
            "13_tb_paren": (-0.065, 0.52, "left"), "12_bk_listidx": (0.205, 0.155, "left"),
            "12_ij2_lang": (0.43, 0.875, "left"), "18_ij2_lang": (0.40, 0.99, "left"),
            "22_code_lang": (0.31, 0.62, "left"), "12_bk_code": (0.02, 0.955, "left"),
}
    for r in rows:
        if r["key"] in offs:
            tx, ty, ha = offs[r["key"]]
            name = NAMES[r["feat"]] + f" ({r['exp']})"
            ax.annotate(name, (r["mean"], r["stated"]), xytext=(tx, ty), ha=ha, va="center",
                        fontsize=15, color=C["ink"], arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    ax.text(0.375, 0.49, f"Spearman, structural rows:\n{rho_st:.2f} (p = {p_st:.2f}, n = {len(st)})\n"
            f"all {n_rows} rows: {rho_all:.2f} (p = {p_all:.3f})", fontsize=15, color=C["ink"], va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["line"]))
    ax.set_xlim(-0.07, 0.66)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("statement shift (mean of 8 paraphrases; whisker = min to max)")
    ax.set_ylabel("NLA states it\n(share of 320 readouts)")

    checks = [
        {"quantity": "rows", "computed": n_rows, "record": 47, "where": "exp 28 RESULTS", "match": n_rows == 47},
        {"quantity": "Spearman within structural rows (perm p)", "computed": f"{rho_st:.2f} (p={p_st:.3f}, n={len(st)})",
         "record": "0.28 (p = 0.08), n = 41", "where": "exp 28 RESULTS; talk S5",
         "match": round(rho_st, 2) == 0.28 and len(st) == 41},
        {"quantity": "Spearman all rows", "computed": f"{rho_all:.2f} (p={p_all:.3f})", "record": "0.40 (p = 0.006)",
         "where": "exp 28 RESULTS", "match": round(rho_all, 2) == 0.40},
        {"quantity": "rows with paraphrase range > 0.15", "computed": f"{len(rng15)} ({', '.join(r['key'] for r in rng15)})",
         "record": "5 of 47 (RESULTS prediction 1); stats28.json says 8", "where": "exp 28 RESULTS vs stats28.json",
         "match": False},
        {"quantity": "max paraphrase-mean shift over structural rows", "computed": round(max_struct_shift, 3),
         "record": "0.00 to 0.19 on every structural row", "where": "talk S5 update note",
         "match": round(max_struct_shift, 2) <= 0.19},
    ]
    for key, rec in [("13_tb_tablecol", (0.05, 0.41)), ("18_call_arg", (0.05, 0.33)), ("13_tb_paren", (0.00, 0.30)),
                     ("12_bk_listidx", (0.14, 0.25))]:
        r = next(x for x in rows if x["key"] == key)
        checks.append({"quantity": f"counter-row {key}: para-mean shift, stated", "computed": f"{r['mean']:.2f}, {r['stated']:.2f}",
                       "record": f"{rec[0]:.2f}, {rec[1]:.2f}", "where": "exp 28 RESULTS scoring table",
                       "match": abs(r["mean"] - rec[0]) < 0.006 and abs(r["stated"] - rec[1]) < 0.006})
    title = "The statement shift separates language from structure but does not predict what the NLA states within structure"
    caption = (f"Each dot is one of {n_rows} rows from exps 12, 13, 18, 19 and 22: x = statement shift through the NLA "
               "critic, mean over eight paraphrase pairs (horizontal whisker: min to max); y = share of 320 NLA free "
               "readouts (40 items x 8) a judge marked as stating the property (vertical whisker: Wilson 95%, which "
               f"ignores clustering by item). Every structural row has shift ≤ {max_struct_shift:.2f}, yet table column, "
               "call argument and parenthesis are stated at 0.30 to 0.41 with shifts near zero; within structural rows "
               f"Spearman is {rho_st:.2f} (20,000-shuffle permutation p = {p_st:.2f}). Filled = structural rows, open = "
               "self-inducing rows (language, caps, tense); (n) = experiment. The 0.30 rule is shown only as withdrawn.")
    notes = ("Computed from experiments/28_stmt_curve/results/stmt28_*.json (shift, paraphrase min/max, stated) and rows28.json "
             "(type, counter/language flags, n readouts). Record discrepancy: exp 28 RESULTS says the paraphrase range "
             f"exceeds 0.15 on 5 of 47 rows (language rows and tense); the raw files and stats28.json give {len(rng15)}: "
             + ", ".join(f"exp {r['exp']} {style.ROW_WORDS.get(r['feat'], r['feat'])} {r['rng']:.2f}" for r in rng15) + ". Prediction 1 fails either way. Counts k = "
             "round(stated x 320) as in analyze28.py. The exp 19 rows enter only as aggregates (no item text).")
    style.save(fig, "s5_nla_states_vs_shift", title=title, caption=caption, section="S5", role="main",
               sources=["experiments/28_stmt_curve/results/stmt28_*.json", "experiments/28_stmt_curve/results/rows28.json"],
               checks=checks, notes=notes)
    for c in checks:
        print("OK  " if c["match"] else "DIFF", c["quantity"], c["computed"], "| record", c["record"])


if __name__ == "__main__":
    main()
