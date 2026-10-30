"""s6_carry_ladder_slide (exp 20): the carry ladder cut to four series for the slide: the model's own 32-token window,
the same tokens shuffled, Oracle Lens's readout and the NLA's explanation. Same data path as g4_s6_carry_ladder.py;
the full six-series figure is the backup s6_carry_ladder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402
from g4_s6_carry_ladder import ROWS, gather  # noqa: E402

KEEP = [  # key, legend label, colour, marker, filled
    ("model_span_32", "model's own window (last 32 tokens)", C["probe"], "s", True),
    ("bag_32", "the same 32 tokens, shuffled", C["muted"], "X", True),
    ("ol_readout", "Oracle Lens readout", C["ol"], "D", True),
    ("nla_readout", "NLA explanation", C["nla"], "o", True),
]
KEPT_KEYS = {k for k, *_ in KEEP}


def main():
    style.slide_style()
    data, sources, checks_all = gather()
    checks = [c for c in checks_all if c["quantity"].split(" (2dp)")[0].split(" ")[-1] in KEPT_KEYS
              or not c["quantity"].startswith(tuple(ROWS))]

    fig, ax = plt.subplots(figsize=G.FIG, layout="constrained")
    gap = 0.5
    ypos = {r: (i if r in G.MARKDOWN else i + gap) for i, r in enumerate(ROWS[::-1])}
    off = [0.27, 0.09, -0.09, -0.27]
    ax.axhspan(-0.5, 1.5, color=C["line"], alpha=0.30, lw=0, zorder=0)
    for (key, lab, col, mk, filled), dy in zip(KEEP, off):
        for row in ROWS:
            a, lo, hi = data[row][key]
            md = row in G.MARKDOWN
            y = ypos[row] + dy
            ax.plot([lo, hi], [y, y], color=col, alpha=0.35, lw=2, solid_capstyle="butt", zorder=2)
            ax.plot(a, y, mk, ms=8 if mk != "X" else 9, mfc=col if filled else "white", mec=col, mew=1.5,
                    alpha=0.5 if md else 1.0, zorder=3, label=lab if row == ROWS[0] else None)
    ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=1)
    ax.axvline(0.8, color=C["line"], lw=1.5, ls="--", zorder=1)
    top = max(ypos.values()) + 0.5
    ax.text(0.5, top, "chance", ha="center", va="bottom", color=C["muted"], fontsize=15)
    ax.text(0.8, top, "0.80", ha="center", va="bottom", color=C["muted"], fontsize=15)
    ax.set_yticks([ypos[r] for r in ROWS])
    ax.set_yticklabels([G.SHORT[r] for r in ROWS])
    for t, r in zip(ax.get_yticklabels(), ROWS):
        if r in G.MARKDOWN:
            t.set_color(C["muted"])
    ax.set_ylim(-0.55, top + 0.45)
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel("state recovered (frozen-probe AUROC, 0.5 = chance)")
    ax.text(1.0, 0.5, "not linearly\npresent", fontsize=15, color=C["muted"], va="center", ha="right",
            linespacing=1.1)
    fig.legend(loc="outside upper center", ncol=2, handletextpad=0.2, columnspacing=1.0)
    style.save(
        fig, "s6_carry_ladder_slide",
        title="The state is in the model's window and in both readers' words; shuffling the words removes it",
        caption=("Exp 20, 160 selection items per row (80 vs 80): AUROC of a probe fit on the true layer-42 activations, "
                 "applied frozen to the model's own activation over the last 32 tokens and to the critic's rebuild from "
                 "the same 32 tokens shuffled, from Oracle Lens's free readout and from the NLA's explanation (k = 1). "
                 "Oracle Lens is weakest on table column, parentheses and call argument; the two Markdown rows (shaded) "
                 "are not linearly present at layer 42. Bars: Hanley-McNeil 95% (no per-item scores saved; ignores "
                 "clustering). Caveat: the critic is the NLA's own, so the NLA series is in distribution."),
        section="S6", role="main", sources=sorted(set(sources)), checks=checks,
        notes=("Four of the six exp 20 series; the last-4-token window and the critic on the true 32 tokens are in the "
               "backup s6_carry_ladder. 'critic' = the NLA reconstructor (ceselder iter_000400). Readouts: Oracle Lens k = 1 "
               "at layer 42 with the standard carrier; NLA raw explanation k = 1 with the actor prompt in "
               "notes/wsbench_internals.md ('You are a meticulous AI researcher ... <concept>...</concept> Please provide "
               "an explanation.'). Rows 'table column' and 'inside parentheses' are exp 13's banks, the others exp 19's "
               "(aggregate numbers only)."),
    )
    print("checks", len(checks), "mismatch", [c for c in checks if not c["match"]])


if __name__ == "__main__":
    main()
