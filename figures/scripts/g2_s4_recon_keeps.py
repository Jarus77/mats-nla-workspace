"""s4_recon_keeps (exp 14): what a reconstructor keeps from real text, from Oracle Lens's bullets, from a statement."""
from __future__ import annotations

import json

import g2_common as g
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style
from style import C, save

RES = g.EXP / "14_ar_blindspot" / "results"
ROWS = [r for r, _ in g.S4_ROWS]


def main() -> None:
    A = {r: json.loads((RES / f"analysis14_{r}.json").read_text()) for r in ROWS}
    D = {}
    for r in ROWS:
        L = A[r]["ol"]["layers"]["44"]
        st = L["stmt"]
        D[r] = {"span": L["span32"]["probe_retrained_on_recon"], "bullets": L["ol_bullets"]["probe_from_span32"],
                "bullets_gap": L["ol_bullets"]["gap_vs_span32"]["ratio"],
                "ol_stmt": st[f"ol:ol:{r}:fixed"]["ratio"],
                "ol_best": max((v["ratio"], k) for k, v in st.items() if k.startswith("ol:ol:")),
                "nla_stmt": A[r]["nla"]["stmt"][f"nla:{r}:fixed"]["ratio"]}

    style.slide_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.92, 4.33), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.06})
    fig.subplots_adjust(left=0.2, right=0.985, bottom=0.17, top=0.9)
    for r in ROWS:
        y = g.S4_Y[r]
        d = D[r]
        ax.plot([d["bullets"], d["span"]], [y, y], color=C["line"], lw=2.5, zorder=1)
        ax.plot(d["span"], y, "o", ms=15, color=C["probe"], mec="white", mew=1.2, zorder=3)
        ax.plot(d["bullets"], y, "o", ms=11, color=C["ol"], mec="white", mew=1.2, zorder=3)
        top = r == "tb_tablecol"  # the first row names the two series directly
        ax.text(1.045 if top else d["span"] + 0.012, y + 0.22, (f"true span {d['span']:.2f}" if top else f"{d['span']:.2f}"),
                fontsize=15, color=C["probe"], ha="right" if top else "center")
        ax.text(d["bullets"] - (0.03 if top else 0), y - 0.47, (f"{d['bullets']:.2f} own bullets" if top else f"{d['bullets']:.2f}"),
                fontsize=15, color=C["ol"], ha="left" if top else "center")
    g.s4_rows(ax)
    ax.axvline(0.5, color=C["line"], ls="--", lw=1.5, zorder=0)
    ax.text(0.5, -0.72, "chance", ha="center", va="bottom", fontsize=15, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlim(0.45, 1.05)
    ax.set_xlabel("probe AUROC on the rebuild")
    ax.set_ylim(-0.78, 4.45)
    ax.set_title("A. Oracle Lens's reconstructor fed:", loc="left", x=-0.25)


    # right panel: statement shift
    for r in ROWS:
        y = g.S4_Y[r]
        d = D[r]
        bx.barh(y + 0.17, d["nla_stmt"], height=0.3, color=C["nla"])
        bx.barh(y - 0.17, d["ol_stmt"], height=0.3, color=C["ol"])
        for v, dy, col in ((d["nla_stmt"], 0.17, C["nla"]), (d["ol_stmt"], -0.17, C["ol"])):
            bx.text(max(v, 0) + 0.02, y + dy, f"{v:+.2f}", va="center", fontsize=15, color=col)
    lang = D["ij2_lang"]
    bx.plot(lang["ol_best"][0], g.S4_Y["ij2_lang"] - 0.17, "D", ms=10, mfc="white", mec=C["ol"], mew=2)
    bx.annotate(f"{lang['ol_best'][0]:.2f}: a bullet written\nin French exhibits,\nit does not state",
                xy=(lang["ol_best"][0], g.S4_Y["ij2_lang"] - 0.17), xytext=(0.62, 0.8), fontsize=15, color=C["ol"],
                ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color=C["ol"], lw=1))
    g.s4_rows(bx, labels=False)
    bx.axvline(0, color=C["muted"], lw=1)
    bx.axvline(0.30, color=C["line"], ls="--", lw=1.3, zorder=0)
    bx.text(0.30, -0.72, "0.30", ha="center", va="bottom", fontsize=15, color=C["muted"], bbox=dict(fc="white", ec="none", pad=0.5))
    bx.set_xlim(-0.12, 1.55)
    bx.set_xlabel("share of the class gap moved")
    bx.set_ylim(-0.78, 4.45)
    bx.set_title("B. A one-sentence statement fed to:", loc="left")
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(color=C["nla"], label="NLA's critic"),
                       Patch(color=C["ol"], label="Oracle Lens's\nreconstructor")],
              loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.2, borderaxespad=0.1)

    rec = {"tb_tablecol": (1.00, 0.66, 0.03), "tb_paren": (0.93, 0.60, 0.10), "ij2_lang": (0.91, 0.92, 0.10)}
    checks = []
    for r, (sp, bu, st) in rec.items():
        checks += [g.check(f"{r} Oracle Lens reconstructor, true span, probe AUROC", round(D[r]["span"], 2), sp, "exp 14 RESULTS table"),
                   g.check(f"{r} Oracle Lens reconstructor, Oracle Lens bullets, true-text probe AUROC", round(D[r]["bullets"], 2), bu, "exp 14 RESULTS table"),
                   g.check(f"{r} Oracle Lens reconstructor, fixed descriptive statement, gap share", round(D[r]["ol_stmt"], 2), st, "exp 14 RESULTS table")]
    checks += [
        g.check("language Oracle Lens best bullet statement gap share", round(lang["ol_best"][0], 2), 1.39, "talk script S4 table ('a statement ... 1.39')"),
        {"quantity": "what the talk's 'a statement of the property' language 1.39 is", "computed": "'- Je recommande' vs '- Je empfehle' (m1, a bullet written in the language); the descriptive '- French language' moves 0.10",
         "record": "talk script S4: statement 0.03 / 0.10 vs 1.39 for language", "where": "talk_script.md S4 exp 14 table", "match": False},
        {"quantity": "units of the talk's span / bullets rows", "computed": "probe AUROC (1.00, 0.93 / 0.66, 0.60); as share of class gap, bullets keep 0.12 / 0.18 (language 0.28)",
         "record": "talk script S4 header: 'share of the class gap kept' for all three rows", "where": "talk_script.md S4 exp 14 table", "match": False},
        {"quantity": "parenthesis 0.44 bullet pair", "computed": f"{D['tb_paren']['ol_best'][1]} = '- /or' vs '- /early' ({D['tb_paren']['ol_best'][0]:.3f}); '- ), and' vs '- and' (m1) = -0.169",
         "record": "exp 14 RESULTS: \"the punctuation-bearing '- ), and' vs '- and' (0.44)\"", "where": "exp 14 RESULTS.md point 2", "match": False},
        {"quantity": "which reconstructor the talk's exp 14 numbers come from", "computed": "Oracle Lens's own reconstructor at layer 44 (whitened output space); through the NLA critic the statements give 0.07 / -0.05 / 0.72",
         "record": "talk glossary defines the reconstructor as the NLA's critic and the S4 table does not say which one", "where": "talk_script.md glossary + S4", "match": False},
        g.check("NLA critic fixed statement, table column", round(D["tb_tablecol"]["nla_stmt"], 2), 0.07, "exp 14 RESULTS table"),
        g.check("NLA critic fixed statement, parenthesis", round(D["tb_paren"]["nla_stmt"], 2), -0.05, "exp 14 RESULTS table"),
        g.check("NLA critic fixed statement, language", round(D["ij2_lang"]["nla_stmt"], 2), 0.72, "exp 14 RESULTS table"),
        g.check("NLA critic fixed statement, list item index", round(D["bk_listidx"]["nla_stmt"], 2), 0.23, "exp 14 RESULTS table"),
    ]
    save(fig, "s4_recon_keeps",
         title="Reconstructors rebuild position from real text but not from a sentence that states it",
         caption=("Exp 14, 200 items per row. A: a probe trained on Oracle Lens's reconstructor output for the true 32-token span "
                  "(grouped 5-fold) separates every row; fed Oracle Lens's own bullets it falls to 0.54-0.66 on the position rows but "
                  "not on language. Right: a one-sentence statement moves each reconstructor's output along the true class direction "
                  "by under 0.30 of the gap on every position row; only the language statement through the NLA's critic clears it "
                  "(0.72). Statements are the fixed labels, e.g. 'The text is in French' (NLA critic) and '- French language' (Oracle Lens). Oracle Lens's reconstructor ignores even '- French language' (0.10); its 1.39 comes from a bullet written in "
                  "French. Caveat: aggregates only (no per-item file is local), so no intervals; Oracle Lens reconstructor at layer 44, in its "
                  "whitened output space."),
         section="S4", role="main",
         sources=[g.rel(RES / f"analysis14_{r}.json") for r in ROWS] + ["experiments/07_template_mining/templates_frozen.json",
                                                                        "experiments/13_new_rows/templates_frozen.json"],
         checks=checks,
         notes=("Statements: NLA critic uses the fixed-label sentence of the NLA ('The current cell is in the first/fifth column of the "
                "table', 'The current position is inside/outside a parenthesis', 'This is item number two/nine in the list', 'The text "
                "is in French/German'); Oracle Lens's reconstructor uses its fixed bullet ('- column one/five', '- inside/outside "
                "parentheses', '- item two/nine', '- French/German language'). As a share of the class gap, Oracle Lens bullets keep "
                + ", ".join(f"{r} {D[r]['bullets_gap']:.2f}" for r in ROWS)
                + ". The talk table mixes units (span and bullets rows are probe AUROCs, the statement row a gap share) and pairs a "
                "descriptive statement for position with an exhibiting French bullet for language. No raw per-item reconstructions are "
                "on disk; numbers are read from the analysis14_*.json aggregates."))


if __name__ == "__main__":
    main()
