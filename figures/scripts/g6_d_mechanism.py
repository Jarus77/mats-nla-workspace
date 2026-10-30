"""Diagram: the section-4 mechanism (exps 14, 15): span vs statement through the reconstructor, and who
follows the class direction vs the statement direction. Numbers read from the raw analysis JSON."""
from __future__ import annotations

import json
import math

from g6_common import C, ROOT, SOFT, WHITE, arrow, box, canvas, text
from style import save

NAME = "s4_d_mechanism"
E14 = ROOT / "experiments/14_ar_blindspot/results"
E15 = ROOT / "experiments/15_ar_steering/results"


def load():
    a = {f: json.loads((E14 / f"analysis14_{f}.json").read_text()) for f in ("tb_tablecol", "tb_paren", "ij2_lang")}
    ol = {f: a[f]["ol"]["layers"]["44"] for f in a}
    n14 = {
        "span_auc_col": ol["tb_tablecol"]["span32"]["probe_retrained_on_recon"],
        "span_auc_par": ol["tb_paren"]["span32"]["probe_retrained_on_recon"],
        "stmt_col": ol["tb_tablecol"]["stmt"]["ol:ol:tb_tablecol:fixed"]["ratio"],
        "stmt_par": ol["tb_paren"]["stmt"]["ol:ol:tb_paren:fixed"]["ratio"],
        "lang_fixed": ol["ij2_lang"]["stmt"]["ol:ol:ij2_lang:fixed"]["ratio"],
        "lang_best": max(v["ratio"] for k, v in ol["ij2_lang"]["stmt"].items() if k.startswith("ol:")),
        "nla_stmt_col": a["tb_tablecol"]["nla"]["stmt"]["nla:tb_tablecol:fixed"]["ratio"],
        "nla_stmt_par": a["tb_paren"]["nla"]["stmt"]["nla:tb_paren:fixed"]["ratio"],
        "nla_stmt_lang": a["ij2_lang"]["nla"]["stmt"]["nla:ij2_lang:fixed"]["ratio"],
    }
    b = {f: json.loads((E15 / f"analysis15_{f}.json").read_text()) for f in ("tb_tablecol", "tb_paren")}

    def z(f, d, s):
        return b[f]["cells"][f"{d}:r11"][s]["iso"]["band"]["z"]

    n15 = {
        "cos_col": b["tb_tablecol"]["geometry"]["cosines"]["cm_raw"]["arstmt_raw"],
        "model_cm_col": z("tb_tablecol", "cm_raw", "model_lo_newline"),
        "model_st_col": z("tb_tablecol", "arstmt_raw", "model_lo_newline"),
        "model_cm_par": -z("tb_paren", "cm_raw", "model_lo_close"),  # sign flipped: + = expected
        "model_st_par": -z("tb_paren", "arstmt_raw", "model_lo_close"),
        "ol_cm_col": z("tb_tablecol", "cm_raw", "ol_mean"), "ol_st_col": z("tb_tablecol", "arstmt_raw", "ol_mean"),
        "jl_cm_col": z("tb_tablecol", "cm_raw", "jlens"), "jl_st_col": z("tb_tablecol", "arstmt_raw", "jlens"),
        "model_span_col": z("tb_tablecol", "arspan_raw", "model_lo_newline"),
        "model_span_par": -z("tb_paren", "arspan_raw", "model_lo_close"),
        "model_expl_par": -z("tb_paren", "arexpl_raw", "model_lo_close"),
    }
    return n14, n15


def fz(v):
    return f"{v:+.1f}".replace("-", "−")


def fz2(v):
    return f"{v:.2f}".replace("-", "−")


def main():
    n14, n15 = load()
    fig, ax = canvas()  # 92.4 x 42 units

    # ---- panel 1: what the reconstructor keeps (exp 14) ----
    p1 = 0.5
    text(ax, p1, 40.0, "1  What is rebuilt", fs=16, weight="bold", ha="left")
    text(ax, p1, 35.4, "exp 14 · Oracle Lens's\nreconstructor, layer 44", fs=15, ha="left", color=C["muted"],
         linespacing=1.1)
    box(ax, p1, 25.0, 26, 5.4, "real 32-token span", fc=SOFT, ec=C["probe"], fs=15)
    text(ax, p1 + 0.3, 21.9, "classes still separate:", fs=15, ha="left", color=C["probe"])
    text(ax, p1 + 0.3, 19.0, f"AUROC {n14['span_auc_col']:.2f} / {n14['span_auc_par']:.2f}", fs=16,
         ha="left", color=C["probe"], weight="bold")
    box(ax, p1, 10.6, 26, 5.4, "a statement of it", fc=SOFT, ec=C["text"], fs=15)
    text(ax, p1 + 0.3, 7.5, "moves the rebuild only", fs=15, ha="left", color=C["text"])
    text(ax, p1 + 0.3, 4.6, f"{n14['stmt_col']:.2f} / {n14['stmt_par']:.2f} of the gap", fs=16, ha="left",
         color=C["text"], weight="bold")
    text(ax, p1 + 0.3, 1.4, "(table column / parenthesis)", fs=15, ha="left", color=C["muted"])

    # ---- panel 2: two directions ----
    p2 = 33.0
    text(ax, p2, 40.0, "2  Two directions", fs=16, weight="bold", ha="left")
    text(ax, p2, 35.4, "exp 15 · built by the\nNLA's critic, layer 42", fs=15, ha="left", color=C["muted"],
         linespacing=1.1)
    ox, oy, L = p2 + 1.5, 11.0, 21
    ang = math.acos(n15["cos_col"])
    arrow(ax, (ox, oy), (ox + L, oy), color=C["probe"], lw=4, ms=22)
    text(ax, ox + L / 2, oy - 4.6, "the state\n(class mean)", fs=15, color=C["probe"], weight="bold",
         linespacing=1.1)
    sx, sy = ox + L * 0.9 * math.cos(ang), oy + L * 0.9 * math.sin(ang)
    arrow(ax, (ox, oy), (sx, sy), color=C["text"], lw=4, ms=22)
    text(ax, sx + 1.6, sy - 2.6, "a statement\nof the state", fs=15, color=C["text"], weight="bold", ha="left",
         linespacing=1.1)
    text(ax, ox + 3.0, oy + 3.4, f"cos {n15['cos_col']:.2f}", fs=15, color=C["muted"], ha="left")

    # ---- panel 3: who follows which (exp 15) ----
    p3 = 58.0
    text(ax, p3, 40.0, "3  Who follows which", fs=16, weight="bold", ha="left")
    text(ax, p3, 35.4, "exp 15 · push z score,\ntable column", fs=15, ha="left", color=C["muted"],
         linespacing=1.1)
    cx = [76.6, 86.8]
    text(ax, cx[0], 29.6, "state", fs=15, color=C["probe"])
    text(ax, cx[1], 29.6, "statement", fs=15, color=C["text"])
    rows = [("the model", C["probe"], n15["model_cm_col"], n15["model_st_col"]),
            ("Oracle Lens", C["ol"], n15["ol_cm_col"], n15["ol_st_col"]),
            ("J-lens", C["jl"], n15["jl_cm_col"], n15["jl_st_col"])]
    ys = [24.6, 18.0, 11.4]
    for (lab, col, a, b), y in zip(rows, ys):
        box(ax, p3, y - 2.9, 92.2 - p3, 5.8, fc=SOFT, ec=col, lw=1.6)
        text(ax, p3 + 0.8, y, lab, fs=15, ha="left", color=col, weight="bold")
        strong = [abs(a) >= abs(b), abs(b) > abs(a)]
        for v, x, st in zip((a, b), cx, strong):
            text(ax, x, y, fz(v), fs=17 if st else 15, weight="bold" if st else "normal",
                 color=col if st else C["muted"])
    text(ax, p3, 4.4, f"parenthesis, the model:\nstate {fz(n15['model_cm_par'])}, statement "
         f"{fz(n15['model_st_par'])}", fs=15, color=C["probe"], ha="left", linespacing=1.1)

    checks = [
        {"quantity": "exp 14 true span, table column / parenthesis", "computed":
         f"{n14['span_auc_col']:.4f} / {n14['span_auc_par']:.4f} (probe AUROC on Oracle Lens reconstructor output, layer 44)", "record":
         "talk script S4 table: 1.00 / 0.93 labelled 'share of the class gap kept'; brief: 'keeps 0.93-1.00 of class "
         "gap'", "where": "talk script S4, exp 14 RESULTS", "match": False,
         "note": "values match but they are AUROCs, not gap shares; the record mislabels the unit"},
        {"quantity": "exp 14 lens bullets row", "computed": "probe AUROC 0.66 / 0.60; gap share 0.12 / 0.18",
         "record": "talk script: 0.66 / 0.60 as gap share", "where": "talk script S4", "match": False,
         "note": "AUROC, not gap share (not drawn)"},
        {"quantity": "exp 14 statement shift (Oracle Lens reconstructor, fixed '- column one/five'), column / paren",
         "computed": f"{n14['stmt_col']:.3f} / {n14['stmt_par']:.3f}", "record": "0.03 / 0.10", "where":
         "exp 14 RESULTS, talk script S4", "match": round(n14['stmt_col'], 2) == 0.03 and round(n14['stmt_par'], 2) == 0.10},
        {"quantity": "language 1.39", "computed": f"best Oracle Lens bullet pair (written in French vs German) "
         f"{n14['lang_best']:.3f}; '- French language' statement {n14['lang_fixed']:.3f}", "record":
         "talk script: 'a statement of the property ... language 1.39'", "where": "talk script S4", "match": False,
         "note": "1.39 is bullets written in the language (exhibiting), not a statement of it; the statement is 0.10"},
        {"quantity": "exp 14 reconstructor identity", "computed": "Oracle Lens reconstructor (agu18dec ar_ptag_pooled) at layer 44",
         "record": "talk script S4 says 'the reconstructor' next to a glossary that defines it as the NLA's critic",
         "where": "talk script S4", "match": False},
        {"quantity": "exp 15 model z, class / statement, table column", "computed":
         f"{n15['model_cm_col']:.2f} / {n15['model_st_col']:.2f}", "record": "+7.1 / +0.7", "where": "exp 15 RESULTS",
         "match": round(n15['model_cm_col'], 1) == 7.1 and round(n15['model_st_col'], 1) == 0.7},
        {"quantity": "exp 15 model z, class / statement, parenthesis (sign flipped)", "computed":
         f"{n15['model_cm_par']:.2f} / {n15['model_st_par']:.2f}", "record": "+19.8 / -3.6", "where": "exp 15 RESULTS",
         "match": round(n15['model_cm_par'], 1) == 19.8 and round(n15['model_st_par'], 1) == -3.6},
        {"quantity": "exp 15 Oracle Lens / J-lens z on statement direction, table column", "computed":
         f"{n15['ol_st_col']:.2f} / {n15['jl_st_col']:.2f}", "record": "+12.3 / +10.1", "where": "exp 15 RESULTS",
         "match": round(n15['ol_st_col'], 1) == 12.3 and round(n15['jl_st_col'], 1) == 10.1},
        {"quantity": "exp 15 'rebuilt from the true span' parenthesis model z", "computed":
         f"span {n15['model_span_par']:.2f}; explanations {n15['model_expl_par']:.2f}", "record":
         "talk script S4: +12.9", "where": "talk script S4 exp 15 table", "match": False,
         "note": "+12.9 is the AR(NLA explanations) direction; the span direction is +6.0 (exp 15 RESULTS agrees with raw)"},
        {"quantity": "cos(class mean, statement direction), table column", "computed": f"{n15['cos_col']:.3f}",
         "record": "0.10", "where": "exp 15 RESULTS geometry", "match": round(n15['cos_col'], 2) == 0.10},
    ]
    save(fig, NAME, title="The model follows the state; the readers follow a sentence about it",
         caption=("Panel 1 (exp 14, Oracle Lens's reconstructor at layer 44, 200 items per row): a true 32-token span "
                  "rebuilt by the reconstructor still separates the classes (probe AUROC), while a statement of the "
                  "property moves the rebuild 0.03 to 0.10 of the class gap; text written in the language moves it "
                  "1.39. Panels 2-3 (exp 15, NLA critic, layer 42): pushing h at 11% of its norm along the class "
                  "direction moves the model's next token (z against 20 random directions); the statement direction "
                  "(cos 0.10) does not move the model but is what Oracle Lens and the J-lens follow."),
         section="S4", role="main",
         sources=[str(p.relative_to(ROOT)) for p in [E14 / "analysis14_tb_tablecol.json", E14 / "analysis14_tb_paren.json",
                                                    E14 / "analysis14_ij2_lang.json", E15 / "analysis15_tb_tablecol.json",
                                                    E15 / "analysis15_tb_paren.json"]],
         checks=checks,
         notes=(f"Moved off the slide (computed here): language control, Oracle Lens reconstructor: bullets written in "
                f"French vs German move the rebuild {n14['lang_best']:.2f} of the gap, the statement '- French language' "
                f"vs '- German ...' {n14['lang_fixed']:.2f}; the NLA's critic on the fixed statement: table column "
                f"{n14['nla_stmt_col']:.2f}, parenthesis {fz2(n14['nla_stmt_par'])}, language {n14['nla_stmt_lang']:.2f}. "
                "Pushes are 11% of |h|, z against 20 random directions; 'the model' is its own next-token log-odds "
                "(newline for table column, close-paren for parenthesis). "
                "Exp 14 numbers use the Oracle Lens reconstructor (ar_ptag_pooled, tag 44) in its whitened output space; "
                "the span value is a probe retrained on the rebuilt spans (AUROC), not a gap share. Exp 15 directions are "
                "built with the NLA's critic; parenthesis model z is sign-flipped so + = expected. Statement pair (exp 15 "
                "NLA): 'The current cell is in the first column of the table' vs the fifth-column version; Oracle Lens pair (exp 14): "
                "'- column one' vs '- column five'."))


if __name__ == "__main__":
    main()
