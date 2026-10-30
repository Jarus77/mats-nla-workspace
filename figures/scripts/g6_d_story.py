"""Diagram: the sprint arc as phases, with dropped hypotheses at the date they fell (talk S7 / opener)."""
from __future__ import annotations

from matplotlib.patches import Rectangle

from g6_common import C, canvas, text
from style import save

NAME = "s7_d_story"

PHASES = [  # dates, label, exps, faint
    ("5-6 Oct", "benchmark can't test reading", "exps 01, 06", False),
    ("6-13 Oct", "J-space pilots (superseded)", "superseded by the 23 Oct redirect", True),
    ("22-24 Oct", "instruments on the activation", "exp 12, proposal", False),
    ("22-26 Oct", "position rows: used, not stated", "exps 12, 13, 17", False),
    ("23-25 Oct", "mechanism: span vs statement", "exps 14, 15, 16", False),
    ("26-27 Oct", "pre-registered prediction tests", "exps 18, 19, 22", False),
    ("27-28 Oct", "readers carry it, don't name it", "exps 20, 23-25, 27", False),
    ("28-30 Oct", "layers, training path, adapter", "exps 26, 28, 29, 30", False),
]
DROPPED = {  # phase index -> (date, display lines, [(label, source)])
    5: ("26 Oct", ["'NLA states it when used'"], [("'NLA states it when it is used'", "D42, exp 18")]),
    6: ("27 Oct", ["window account · theory day", "continuation predictor", "loose naming count"],
        [("window account", "D45, exp 20"), ("theory day", "derivation.md"), ("continuation predictor", "exp 21"),
         ("loose naming count", "D47, exp 23")]),
    7: ("28 Oct", ["0.30 statement rule"], [("0.30 statement rule", "D49, exp 28")]),
}


def main():
    fig, ax = canvas()  # 92.4 x 42 units
    xd, xl, xx, xi = 0.5, 13.0, 48.6, 60.2  # date, phase label, drop date, drop items
    pitch = 3.45
    text(ax, xd, 40.0, "phase", fs=16, weight="bold", ha="left", color=C["probe"])
    text(ax, xx, 40.0, "six ideas dropped (log date)", fs=16, weight="bold", ha="left", color=C["nla"])
    y = 35.6
    ax.plot([0.5, 92.2], [y + 2.0, y + 2.0], color=C["line"], lw=1.2)
    for i, (dates, lab, _exps, faint) in enumerate(PHASES):
        drop = DROPPED.get(i)
        nlines = len(drop[1]) if drop else 1
        col = C["muted"] if faint else C["ink"]
        yb = y - (nlines - 1) * pitch - 1.75
        if drop:
            ax.add_patch(Rectangle((xx - 1.0, yb), 92.2 - xx + 1.0, y + 1.75 - yb, fc="#FBE9E4", ec="none", zorder=0))
        yt = y - 0.75  # baseline
        text(ax, xd, yt, dates, fs=15, weight="bold", ha="left", va="baseline",
             color=C["muted"] if faint else C["probe"])
        text(ax, xl, yt, lab, fs=15, ha="left", va="baseline", color=col, style="italic" if faint else "normal")
        if drop:
            text(ax, xx, yt, f"✗ {drop[0]}", fs=15, weight="bold", ha="left", va="baseline", color=C["nla"])
            for k, line in enumerate(drop[1]):
                text(ax, xi, yt - k * pitch, line, fs=15, ha="left", va="baseline", color=C["nla"])
        ax.plot([0.5, 92.2], [yb, yb], color=C["line"], lw=1.0)
        y = yb - 1.75
    checks = [
        {"quantity": "window account retired", "computed": "27 Oct (decisions.md D45)", "record": "talk script S7: "
         "'the window account (26 Oct)'", "where": "talk script S7 table", "match": False},
        {"quantity": "'NLA states when it is used' retired", "computed": "26 Oct (D42, exp 18 scored; rule rewritten)",
         "record": "talk script S7: exp 18 rule", "where": "notes/decisions.md", "match": True},
        {"quantity": "theory day dropped", "computed": "dropped by 27 Oct (notes/derivation.md written 27 Oct; "
         "proposed 24 Oct)", "record": "talk script S7 (no date)", "where": "notes/derivation.md", "match": True},
        {"quantity": "continuation predictor (exp 21)", "computed": "27 Oct (git date of exp 21 RESULTS)",
         "record": "talk script S7 (no date)", "where": "experiments/21_continuation", "match": True},
        {"quantity": "loose naming count", "computed": "27 Oct (D47)", "record": "talk script S7", "where":
         "notes/decisions.md", "match": True},
        {"quantity": "0.30 statement rule withdrawn", "computed": "28 Oct (D49)", "record": "talk script S7: 28 Oct",
         "where": "notes/decisions.md", "match": True},
        {"quantity": "phase dates", "computed": "first/last git commit dates of each experiment folder and "
         "decisions.md headings (D1-D50)", "record": "-", "where": "git log, notes/decisions.md", "match": True},
    ]
    save(fig, NAME, title="Six ideas fell on the way from benchmark to answer",
         caption=("The sprint as phases, top to bottom; the italic grey row is the J-space pilot work superseded on "
                  "23 Oct. Red entries are the six hypotheses dropped, on the date the decision log records, beside the "
                  "phase that dropped them."),
         section="S7", role="main", sources=["notes/decisions.md", "notes/derivation.md", "git log (experiment folders)"],
         checks=checks, notes=("Diagram, no data. Dates from decisions.md headings and git commit dates. Experiments "
                                "behind each phase (moved off the slide): " + "; ".join(
                                    f"{d} {lab}: {e}" for d, lab, e, _ in PHASES) + ". Sources of each drop: " + "; ".join(
                                    f"{lab} ({src})" for _, _, items in DROPPED.values() for lab, src in items) + "."))


if __name__ == "__main__":
    main()
