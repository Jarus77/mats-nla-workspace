"""Diagram: the instruments (activation side) and the reader scores they predict (talk S2)."""
from __future__ import annotations

from g6_common import C, SOFT, WHITE, arrow, box, canvas, text
from g6_d_readers import activation_icon
from style import save

NAME = "s2_d_instruments"


def main():
    fig, ax = canvas()  # 92.4 x 42 units

    # ---- left: instruments on the activation alone ----
    text(ax, 0.5, 40.4, "On the activation alone", fs=16, weight="bold", color=C["probe"], ha="left")
    inst = [  # name, how, threshold, colour, withdrawn
        ("readable", "linear probe on h", "AUROC ≥ 0.80", C["probe"], False),
        ("span-sayable", "critic rebuilds h from real text", "AUROC ≥ 0.80", C["probe"], False),
        ("statement shift", "critic on a sentence (withdrawn)", "≥ 0.30", C["probe"], True),
        ("used", "push h; the next token moves", "|z| ≥ 3", C["probe"], False),
        ("J-enriched", "direction inside the J-space", "≥ 1.5", C["jl"], False),
    ]
    x0, w, hgt, gap = 0.5, 37.0, 6.6, 0.85
    top = 37.6
    for i, (name, how, thr, col, gone) in enumerate(inst):
        y0 = top - (i + 1) * hgt - i * gap
        box(ax, x0, y0, w, hgt, fc=SOFT, ec=col, lw=1.6)
        text(ax, x0 + 0.9, y0 + hgt - 2.6, name, fs=15, weight="bold", color=col, ha="left", va="baseline")
        text(ax, x0 + w - 0.9, y0 + hgt - 2.6, thr, fs=15, weight="bold", ha="right", va="baseline",
             color=C["muted"] if gone else col)
        text(ax, x0 + 0.9, y0 + 1.7, how, fs=15, ha="left", color=C["muted"] if gone else C["ink"],
             style="italic" if gone else "normal")

    # ---- arrow ----
    arrow(ax, (x0 + w + 0.6, 20.5), (45.6, 20.5), color=C["ink"], lw=2.6, ms=20)
    text(ax, (x0 + w + 45.8) / 2, 23.3, "predict", fs=15, color=C["ink"])

    # ---- right: reader scores as a small table of pre-registered lines ----
    text(ax, 46.6, 40.4, "Reader scores (pre-registered lines)", fs=16, weight="bold", color=C["ink"], ha="left")
    cols = [("NLA", C["nla"], 66.6), ("Oracle\nLens", C["ol"], 76.6), ("J-lens", C["jl"], 86.8)]
    for lab, col, cx in cols:
        text(ax, cx, 34.6, lab, fs=15, weight="bold", color=col, linespacing=1.05)
    rows = [  # label, NLA, Oracle Lens, J-lens
        ("prompted\n(AUROC)", "≥ 0.85", "≥ 0.70", "≥ 0.70"),
        ("states\n(judge)", "≥ 20%", "≥ 5%", None),
        ("exhibits\n(judge)", None, "≥ 30%", None),
        ("complement\n(multi-choice)", "≥ 20%", None, None),
    ]
    rh, ry0 = 7.4, 31.0
    for i, (lab, *cells) in enumerate(rows):
        y1 = ry0 - i * rh
        yc = y1 - rh / 2
        ax.plot([46.6, 92.2], [y1, y1], color=C["line"], lw=1.2, zorder=1)
        text(ax, 46.6, yc, lab, fs=15, ha="left", linespacing=1.05)
        for (_, col, cx), v in zip(cols, cells):
            if v is None:
                text(ax, cx, yc, "–", fs=15, color="#9FB0BA")
            else:
                box(ax, cx - 4.4, yc - 2.0, 8.8, 4.0, v, fc=WHITE, ec=col, lw=1.6, fs=15, color=col,
                    radius=1.2)
    ax.plot([46.6, 92.2], [ry0 - 4 * rh, ry0 - 4 * rh], color=C["line"], lw=1.2, zorder=1)

    sources = ["notes/prereg_prediction_test.md", "notes/decisions.md"]
    checks = [
        {"quantity": "thresholds readable / span / statement / used / J-enriched", "computed":
         "0.80 / 0.80 / 0.30 / |z| >= 3 / 1.5 (prereg_prediction_test.md section 2)", "record":
         "talk script glossary: 0.80 / 0.80 / 0.30 / (blank) / 1.5", "where": "talk script glossary", "match": True},
        {"quantity": "span-sayable instrument", "computed": "frozen-probe AUROC of NLA-critic rebuild of the last 32 "
         "tokens (clarified 26 Oct; gap-ratio form dropped)", "record": "glossary: same", "where": "prereg 2",
         "match": True},
        {"quantity": "reader-score thresholds", "computed": "NLA prompted >= 0.85; Oracle Lens, J-lens prompted >= 0.70; NLA states "
         ">= 20%; Oracle Lens states >= 5%; Oracle Lens exhibits >= 30%; complement >= 20%", "record": "prereg 3", "where":
         "notes/prereg_prediction_test.md", "match": True},
        {"quantity": "0.30 statement rule status", "computed": "withdrawn 28 Oct (D49)", "record": "talk script S5 "
         "update note: withdrawn after exp 28", "where": "notes/decisions.md D49", "match": True},
        {"quantity": "language control numbers (footer removed; now in notes and in s2_language_control)", "computed": "not recomputed (diagram); taken from talk "
         "script S2", "record": "readable 1.00, shift 0.51, NLA states 81-95%", "where": "talk script S2",
         "match": True},
    ]
    save(fig, NAME, title="Measures on the activation alone predict four reader scores",
         caption=("Left: each row (two values of one property) is measured on the activation alone: readable "
                  "(logistic probe, 5-fold grouped by source file), span-sayable (the NLA's critic rebuilds h from the "
                  "true last 32 tokens; the frozen probe scores the rebuild) and statement shift (the critic on a "
                  "one-sentence statement pair, as a share of the class gap), used (a push along the class direction "
                  "at 11% of |h|, z against 20 random directions), J-enriched (share of the class direction in the "
                  "rank-1,536 J-space vs random). Right: the pre-registered line for each reader score: prompted "
                  "forced choice (log-odds AUROC), states (the judge's verdict that a free readout states the "
                  "property), exhibits (the judge finds the value itself written, e.g. text in French), complement "
                  "(multi-candidate argmax picks the unmarked value); a dash means the cell is not scored. The 0.30 "
                  "statement rule was withdrawn on 28 Oct after exp 28."),
         section="S2", role="main", sources=sources, checks=checks,
         notes=("Diagram, no data. The pre-registration calls the complement cell 'NLA names the complement'; it is "
                "scored by a multi-candidate argmax, not by a regex, so the figure labels it 'complement (multi-choice)'. "
                "Language control row (talk script S2, not recomputed here; drawn from raw files in "
                "s2_language_control): readable 1.00, statement shift 0.51, J-enriched, NLA states 81-95%, Oracle "
                "Lens exhibits."))


if __name__ == "__main__":
    main()
