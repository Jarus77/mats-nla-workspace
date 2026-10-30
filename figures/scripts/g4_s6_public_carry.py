"""s6_public_carry (exp 25): on the independent public bank both readers' readouts carry the structural state through
the critic; shuffling removes structure but not language."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402

E25 = G.EXP / "25_public_exhibits" / "results"
ROWS = ["code_lang", "py_string", "js_string", "latex_math", "json_depth", "py_arg", "md_heading", "md_list"]
SRC = [("span_32", "critic on the true last 32 tokens", C["text"], "^"),
       ("bag_32", "same 32 tokens shuffled", C["muted"], "X"),
       ("ol_readout", "Oracle Lens readout", C["ol"], "D"),
       ("nla_readout", "NLA explanation", C["nla"], "o")]
REC = {"py_string": (0.84, 0.46, 0.86, 0.99), "js_string": (0.77, 0.46, 0.84, 0.97), "latex_math": (0.85, 0.50, 0.87, 1.00),
       "py_arg": (0.88, 0.49, 0.51, 0.93), "json_depth": (0.92, 0.76, 0.97, 0.99), "md_heading": (0.62, 0.55, 0.65, 0.70),
       "md_list": (0.50, 0.44, 0.48, 0.49), "code_lang": (0.99, 0.98, 0.98, 0.98)}


def gather():
    """Per row and source: (AUROC, lo, hi) of the critic's rebuild on the public bank (exp 25)."""
    sources, checks, data = [], [], {}
    for row in ROWS:
        f = E25 / f"recon25_{row}.json"
        sources.append(G.rel(f))
        a = G.load(f)["auroc"]
        data[row] = {k: (a[k], *G.hanley_mcneil(a[k], 80, 80)) for k, *_ in SRC}
        for (k, *_), r in zip(SRC, REC[row]):
            checks.append(G.check(f"{row} {k} (2dp)", round(a[k], 2), r, "exp25 RESULTS.md table"))
    nat = ["py_string", "js_string", "latex_math", "json_depth"]
    checks.append(G.check("Oracle Lens readout min over string/math/depth rows (2dp)", round(min(data[r]["ol_readout"][0] for r in nat), 2), 0.84, "talk script '0.84 to 0.97'"))
    checks.append(G.check("Oracle Lens readout max over string/math/depth rows (2dp)", round(max(data[r]["ol_readout"][0] for r in nat), 2), 0.97, "talk script '0.84 to 0.97'"))
    checks.append(G.check("Oracle Lens readout call argument (2dp)", round(data["py_arg"]["ol_readout"][0], 2), 0.51, "talk script '(call argument 0.51)'"))
    return data, sources, checks


def main():
    data, sources, checks = gather()

    fig, ax = plt.subplots(figsize=style.FULL)
    ypos = {}
    y = 0
    for r in ROWS[::-1]:
        ypos[r] = y
        y += 1 if r not in ("md_heading",) else 1.6
        if r == "py_string":
            y += 0.6
    ax.axhspan(-0.5, 1.5, color=C["line"], alpha=0.25, lw=0, zorder=0)
    off = [0.24, 0.08, -0.08, -0.24]
    for (k, lab, col, mk), dy in zip(SRC, off):
        for row in ROWS:
            a, lo, hi = data[row][k]
            md = row in G.MARKDOWN
            ax.plot([lo, hi], [ypos[row] + dy] * 2, color=col, alpha=0.3, lw=2, zorder=1)
            ax.plot(a, ypos[row] + dy, mk, ms=9.5, color=col, alpha=0.5 if md else 1, zorder=3,
                    label=lab if row == ROWS[0] else None)
    ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=0)
    ax.axvline(0.8, color=C["line"], lw=1.5, ls="--", zorder=0)
    top = max(ypos.values()) + 0.5
    ax.text(0.5, top, "chance", ha="center", va="bottom", color=C["muted"], fontsize=12)
    ax.text(0.8, top, "0.80", ha="center", va="bottom", color=C["muted"], fontsize=12)
    ax.set_ylim(-0.6, top + 0.4)
    ax.set_xlim(0.3, 1.02)
    ax.set_yticks([ypos[r] for r in ROWS])
    ax.set_yticklabels([G.ROW_LABEL[r] for r in ROWS])
    for t, r in zip(ax.get_yticklabels(), ROWS):
        if r in G.MARKDOWN:
            t.set_color(C["muted"])
        if r == "code_lang":
            t.set_fontstyle("italic")
    ax.set_xlabel("state recovered through the critic, frozen-probe AUROC (0.5 = chance)")
    ax.annotate("language survives shuffling (0.98);\nstructure does not (0.44 to 0.50, JSON 0.76)",
                (data["code_lang"]["bag_32"][0] - 0.01, ypos["code_lang"] + 0.08), xytext=(0.31, ypos["code_lang"] - 0.2),
                fontsize=12, color=C["muted"], va="center", arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1))
    ax.text(0.305, ypos["py_arg"] - 0.42, "call argument: in the window (0.88), not in what Oracle Lens writes (0.51)",
            fontsize=11.5, color=C["ol"], va="center")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.02), ncol=4, fontsize=12.5, columnspacing=1.0, handletextpad=0.3)
    style.save(
        fig, "s6_public_carry",
        title="On an independent public bank the readers' outputs carry string, math and depth state too",
        caption=("Exp 25 runs exp 20's protocol on the exp 22 public bank (160 selection items per row, 80 vs 80, k = 1 readouts): "
                 "frozen-probe AUROC of the critic's rebuild from each source against the true layer-42 vectors. Oracle Lens's "
                 "readout carries string, math and JSON-depth state at 0.84 to 0.97 (call argument 0.51); the shuffled span keeps "
                 "the language control and loses structure. Bars: Hanley-McNeil 95% (no per-item scores saved; ignores "
                 "clustering). Caveat: the critic is the NLA's own, so the NLA column is in distribution."),
        section="S6", role="backup", sources=sources, checks=checks,
        notes="Public bank built from package directories and arXiv e-prints (exp 22); code language = Python vs JavaScript positive control.",
    )
    print("mismatch", [c for c in checks if not c["match"]])


if __name__ == "__main__":
    main()
