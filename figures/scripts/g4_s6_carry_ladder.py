"""s6_carry_ladder (exp 20): frozen-probe AUROC per row across six sources of text/activation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402

E20 = G.EXP / "20_ar_structure" / "results"
ROWS = ["py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "tb_tablecol", "tb_paren",
        "md_heading", "md_list"]
MARKED = {"py_string": "inside", "js_string": "inside", "latex_math": "inside", "py_arg": "third", "json_depth": "deep",
          "html_depth": "deep", "md_heading": "deep", "md_list": "nested", "tb_tablecol": "first", "tb_paren": "inside"}
SRC = [  # key, label, colour, marker, filled
    ("model_span_4", "model's own activation, last 4 tokens", C["probe"], "s", False),
    ("model_span_32", "model's own activation, last 32 tokens", C["probe"], "s", True),
    ("span_32", "critic on the true last 32 tokens", C["text"], "^", True),
    ("bag_32", "critic on the same 32 tokens shuffled", C["muted"], "X", True),
    ("ol_readout", "critic on Oracle Lens's readout", C["ol"], "D", True),
    ("nla_readout", "critic on the NLA's explanation", C["nla"], "o", True),
]
RECORD = {  # talk script section 6 table (2 dp): model4, model32, critic32, bag, OL, NLA
    "tb_tablecol": [0.87, 1.00, 0.95, 0.46, 0.64, 1.00], "tb_paren": [0.71, 0.90, 0.74, 0.57, 0.67, 0.87],
    "py_string": [0.99, 1.00, 0.85, 0.42, 0.91, 0.99], "js_string": [0.96, 0.99, 0.82, 0.48, 0.90, 0.98],
    "latex_math": [0.99, 0.99, 0.83, 0.46, 0.87, 0.98], "py_arg": [0.97, 0.96, 0.89, 0.50, 0.70, 0.97],
    "json_depth": [0.77, 0.96, 0.87, 0.65, 0.94, 0.98], "html_depth": [0.83, 0.94, 0.74, 0.30, 0.79, 0.93],
    "md_heading": [0.54, 0.57, 0.51, 0.47, 0.59, 0.59], "md_list": [0.58, 0.62, 0.64, 0.55, 0.62, 0.63],
}


def counts(row):
    base = G.EXP / ("13_new_rows" if row.startswith("tb_") else "19_natural_bank") / "results" / "items_base.json"
    sel = [it for it in G.load(base) if it["feature"] == row and it["split"] == "select"]
    n1 = sum(it["value"] == MARKED[row] for it in sel)
    return n1, len(sel) - n1, G.rel(base)


def gather():
    """Per-row AUROC with Hanley-McNeil CIs for every source, plus the record checks."""
    sources, checks, data = [], [], {}
    for row in ROWS:
        rec = G.load(E20 / f"recon20_{row}.json")
        sources.append(G.rel(E20 / f"recon20_{row}.json"))
        n1, n0, src = counts(row)
        sources.append(src)
        assert rec["n"] == n1 + n0
        data[row] = {}
        for (key, *_), r in zip(SRC, RECORD[row]):
            a = rec["auroc"][key]
            data[row][key] = (a, *G.hanley_mcneil(a, n1, n0))
            checks.append(G.check(f"{row} {key} (2dp)", round(a, 2), r, "talk script S6 table / exp20 RESULTS.md"))
    readable = ROWS[:8]
    m32 = [data[r]["model_span_32"][0] for r in readable]
    bag = [data[r]["bag_32"][0] for r in readable]
    checks.append(G.check("model window 32, min over 8 readable rows (2dp)", round(min(m32), 2), 0.90, "talk script 'carries every readable state at 0.90 to 1.00'"))
    checks.append(G.check("shuffled bag, max over 8 readable rows (2dp)", round(max(bag), 2), 0.65, "exp20 RESULTS.md 'fall to 0.30 to 0.65'"))
    return data, sources, checks


def main():
    data, sources, checks = gather()

    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    ypos = {r: i for i, r in enumerate(ROWS[::-1])}
    ypos = {r: (y if r in G.MARKDOWN else y + 0.6) for r, y in ypos.items()}
    off = np.linspace(-0.30, 0.30, len(SRC))
    ax.axhspan(-0.5, 1.5, color=C["line"], alpha=0.25, lw=0, zorder=0)
    for j, (key, lab, col, mk, filled) in enumerate(SRC):
        for row in ROWS:
            a, lo, hi = data[row][key]
            md = row in G.MARKDOWN
            y = ypos[row] - off[j]
            ax.plot([lo, hi], [y, y], color=col, alpha=0.35, lw=2, solid_capstyle="butt", zorder=2)
            ax.plot(a, y, mk, ms=9 if mk != "X" else 10, mfc=col if filled else "white", mec=col, mew=1.8,
                    alpha=0.55 if md else 1.0, zorder=3, label=lab if row == ROWS[0] else None)
    ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=1)
    ax.axvline(0.8, color=C["line"], lw=1.5, ls="--", zorder=1)
    top = max(ypos.values()) + 0.55
    ax.text(0.5, top, "chance", ha="center", va="bottom", color=C["muted"], fontsize=12)
    ax.text(0.8, top, "0.80", ha="center", va="bottom", color=C["muted"], fontsize=12)
    ax.set_yticks([ypos[r] for r in ROWS])
    ax.set_yticklabels([G.ROW_LABEL[r] for r in ROWS])
    for t, r in zip(ax.get_yticklabels(), ROWS):
        if r in G.MARKDOWN:
            t.set_color(C["muted"])
    ax.set_ylim(-0.6, top + 0.35)
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("state recovered, frozen-probe AUROC against the true layer-42 vectors (0.5 = chance)")
    ax.text(0.205, 0.5, "Markdown: not linearly\npresent at layer 42;\nnear chance everywhere",
            fontsize=11.5, color=C["muted"], va="center")
    yb = ypos["html_depth"] - off[3]
    ax.annotate("shuffling the same 32 tokens\ndrops every row toward chance",
                (data["html_depth"]["bag_32"][0], yb - 0.12), xytext=(0.27, 1.95), fontsize=12.5, va="center",
                color=C["muted"], arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1.1))
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.04), ncol=3, fontsize=12.5, handletextpad=0.3, columnspacing=1.2)
    style.save(
        fig, "s6_carry_ladder",
        title="The state is in the model's own window and in the readers' outputs (Oracle Lens weakest on column, parenthesis, argument); shuffling the words removes it",
        caption=("Frozen-probe AUROC (probe fit on true layer-42 activations, applied to each rebuild) per row and source, "
                 "exp 20, 160 selection items per row (80 vs 80). The model's own activation on the last 32 tokens carries "
                 "every readable row at 0.90 to 1.00; the critic on the same tokens shuffled falls toward chance; both "
                 "readers' free readouts (k = 1) re-induce the state through the critic. Bars: Hanley-McNeil 95% intervals "
                 "(no per-item scores were saved; ignores clustering). Caveat: the critic is the NLA's own, so the NLA "
                 "column is in distribution; compare each column with its own baselines."),
        section="S6", role="backup", sources=sorted(set(sources)), checks=checks,
        notes=("Rows 'table column' and 'inside parentheses' are exp 13's banks, the others exp 19's (aggregate numbers only). "
               "'critic' = the NLA reconstructor (ceselder iter_000400). Readers' readouts: Oracle Lens k = 1 at layer 42 with the "
               "standard carrier; NLA raw explanation k = 1 with the actor prompt in notes/wsbench_internals.md."),
    )
    bad = [c for c in checks if not c["match"]]
    print("checks", len(checks), "mismatch", bad)


if __name__ == "__main__":
    main()
