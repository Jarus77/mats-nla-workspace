"""s6_second_decoder (exp 24): through Oracle Lens's own reconstructor (never trained on either reader's output),
Oracle Lens's readouts land on the true span's direction; the NLA's explanations carry the state only for a probe
trained on their own rebuilds."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g4_common as G  # noqa: E402
import style  # noqa: E402
from style import C, plt  # noqa: E402
from g4_s6_carry_ladder import ROWS, counts  # noqa: E402

E24 = G.EXP / "24_second_decoder" / "results"
REC = {  # exp 24 RESULTS.md table, layer 44: frozen (cv) for ol_readout and nla_readout, bag_32 frozen
    "py_string": (0.80, 0.97, 0.56, 0.89, 0.55), "js_string": (0.80, 0.92, 0.51, 0.81, 0.44),
    "latex_math": (0.78, 0.93, 0.44, 0.81, 0.59), "py_arg": (0.61, 0.58, 0.52, 0.59, 0.53),
    "json_depth": (0.74, 0.92, 0.51, 0.83, 0.71), "html_depth": (0.73, 0.78, 0.44, 0.90, 0.74),
    "md_heading": (0.47, 0.57, 0.44, 0.60, 0.51), "md_list": (0.50, 0.68, 0.46, 0.49, 0.53),
    "tb_tablecol": (0.57, 0.71, 0.47, 0.86, 0.62), "tb_paren": (0.60, 0.77, 0.69, 0.77, 0.64),
}


def gather():
    """Per row: [(OL frozen), (OL cv), (NLA frozen), (NLA cv), (bag frozen)] each (AUROC, lo, hi), layer 44."""
    sources, checks, data = [], [], {}
    for row in ROWS:
        f = E24 / f"recon24_{row}.json"
        sources.append(G.rel(f))
        L = G.load(f)["layers"]["44"]
        n1, n0, src = counts(row)
        sources.append(src)
        v = (L["frozen_auroc"]["ol_readout"], L["cv_auroc"]["ol_readout"], L["frozen_auroc"]["nla_readout"],
             L["cv_auroc"]["nla_readout"], L["frozen_auroc"]["bag_32"])
        data[row] = [(a, *G.hanley_mcneil(a, n1, n0)) for a in v]
        for name, a, r in zip(("Oracle Lens frozen", "Oracle Lens cv", "NLA frozen", "NLA cv", "bag frozen"), v, REC[row]):
            checks.append(G.check(f"{row} {name} layer 44 (2dp)", round(a, 2), r, "exp24 RESULTS.md table"))
    six = ROWS[:6]
    olf = sorted(data[r][0][0] for r in six)
    checks.append(G.check("Oracle Lens frozen, five of six readable rows >= 0.73 (min of top five, 2dp)", round(olf[1], 2), 0.73, "talk script S6 update '0.73 to 0.80 on five of six'"))
    checks.append(G.check("Oracle Lens frozen call argument (2dp)", round(data["py_arg"][0][0], 2), 0.61, "talk script S6 update"))
    nlacv = sorted(data[r][3][0] for r in six if r != "py_arg")
    checks.append(G.check("NLA cv min over five readable rows (2dp)", round(nlacv[0], 2), 0.81, "talk script '0.81 to 0.90'"))
    checks.append(G.check("NLA cv max over five readable rows (2dp)", round(nlacv[-1], 2), 0.90, "talk script '0.81 to 0.90'"))
    return data, sources, checks


def main():
    data, sources, checks = gather()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    ypos = {r: (i if r in G.MARKDOWN else i + 0.6) for i, r in enumerate(ROWS[::-1])}
    panels = (("lands on the true span's direction\n(probe fit on true-span rebuilds, frozen)", 0, 2),
              ("separable by a probe of its own\n(probe fit on the readout's rebuilds)", 1, 3))
    for ax, (title, io, inl) in zip(axes, panels):
        ax.axhspan(-0.5, 1.5, color=C["line"], alpha=0.25, lw=0, zorder=0)
        ax.axvline(0.5, color=C["line"], lw=1.5, ls="--", zorder=0)
        ax.axvline(0.8, color=C["line"], lw=1.5, ls="--", zorder=0)
        for row in ROWS:
            md = row in G.MARKDOWN
            y = ypos[row]
            series = [(data[row][io], C["ol"], "D", 0.13), (data[row][inl], C["nla"], "o", -0.13)]
            if io == 0:
                series.append((data[row][4], C["muted"], "X", 0.0))
            for (a, lo, hi), col, mk, dy in series:
                ax.plot([lo, hi], [y + dy] * 2, color=col, alpha=0.3, lw=2, zorder=1)
                ax.plot(a, y + dy, mk, ms=9.5 if mk != "X" else 9, color=col, alpha=0.5 if md else 1, zorder=3)
        ax.set_title(title, fontsize=14.5, loc="left")
        ax.set_xlim(0.3, 1.02)
        ax.set_xlabel("AUROC (0.5 = chance)")
        top = max(ypos.values()) + 0.55
        ax.text(0.5, top, "chance", ha="center", va="bottom", color=C["muted"], fontsize=11.5)
        ax.text(0.8, top, "0.80", ha="center", va="bottom", color=C["muted"], fontsize=11.5)
        ax.set_ylim(-0.6, top + 0.4)
    axes[0].text(0.31, 1.88, "NLA: at chance along the span's direction;\nOracle Lens 0.73 to 0.80 on five rows", fontsize=12, va="center", color=C["ink"])
    axes[1].text(0.31, 1.88, "NLA: 0.81 to 0.90 by a probe of its own", fontsize=12, va="center", color=C["nla"])
    axes[0].set_yticks([ypos[r] for r in ROWS])
    axes[0].set_yticklabels([G.ROW_LABEL[r] for r in ROWS])
    for t, r in zip(axes[0].get_yticklabels(), ROWS):
        if r in G.MARKDOWN:
            t.set_color(C["muted"])
    hs = [axes[0].plot([], [], "D", ms=9, color=C["ol"])[0], axes[0].plot([], [], "o", ms=9, color=C["nla"])[0],
          axes[0].plot([], [], "X", ms=9, color=C["muted"])[0]]
    fig.legend(hs, ["Oracle Lens readout", "NLA explanation", "true 32 tokens shuffled (reference)"], loc="lower center",
               bbox_to_anchor=(0.55, 0.99), ncol=3, fontsize=13)
    style.save(
        fig, "s6_second_decoder",
        title="A second decoder recovers the state from Oracle Lens's readouts; the NLA's are separable only by a probe fit on their own rebuilds",
        caption=("Exp 24: each readout (k = 1, exp 20's) is fed to Oracle Lens's own activation reconstructor (trained on text "
                 "spans, layer tag 44), 160 items per row. Left: a probe fit on the rebuilds of the true 32-token span, applied "
                 "frozen: Oracle Lens's readouts reach 0.73 to 0.80 on five of six readable rows (call argument 0.61); the NLA's "
                 "explanations sit at chance. Right: a probe fit by grouped 5-fold CV on each readout's own rebuilds: the NLA's "
                 "explanations separate at 0.81 to 0.90. Bars: Hanley-McNeil 95% (no per-item scores saved; ignores clustering)."),
        section="S6", role="backup", sources=sorted(set(sources)), checks=checks,
        notes=("Reading: the NLA's description lands in a region of the decoder's output unlike the span's image (class-direction "
               "cosine about 0) yet the two classes sit apart there; 'recoverable from the explanation' holds under both decoders, "
               "'re-induces the state as the span would' is a property of the NLA's own critic. Layer 40 within 0.04 (table24.md)."),
    )
    print("mismatch", [c for c in checks if not c["match"]])


if __name__ == "__main__":
    main()
