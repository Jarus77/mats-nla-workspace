"""s3_concept_table: exp 12's 14 concept rows as a table. Asked = prompted AUROC per reader; unprompted = share of the
320 free readouts per concept that the judge says state the value (NLA, Oracle Lens) or whose own text exhibits it
(Oracle Lens). Every number is recomputed from the raw score/judged files via g2_common, then checked against the
bk_exp12_recall manifest and exp 12's combined.json."""
from __future__ import annotations

import json

import g2_common as g
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import style
from style import C, save

GROUPS = [
    ("position\nin text", [("nj1_col", "line position"), ("nj1_col_break", "line break coming"),
                           ("bk_wordpos", "words into the sentence"), ("bk_listidx", "list item index"),
                           ("bk_indent", "indentation level"), ("bk_quote", "inside a quotation")]),
    ("next\nword", [("nj3_predpos", "next word's part of speech"), ("agr_gender", "upcoming noun's gender"),
                        ("agr_number", "upcoming noun's number")]),
    ("controls", [("nj2_tense", "tense"), ("bk_caps", "all caps or not"), ("ij_listnum", "'Item N' number"),
                  ("ij2_lang", "French or German"), ("bk_code", "code language")]),
]
FS = 13            # table text; a table may go to 13 pt at slide size
FS_HEAD = 13.5
READABLE = 0.80    # declared readable threshold
NEAR_CHANCE = 0.60
TINT = 0.20        # alpha of the reader-colour cell tint
GREY = "#A3AEB5"   # light grey text for near-chance / zero cells

# column x edges (inches on a 9.22 in canvas)
X_FAM, X_ROW = 0.0, 0.98
COLS = [  # (key, reader colour key, kind)
    ("p_nla", "nla", "auroc"), ("p_ol", "ol", "auroc"), ("p_jl", "jl", "auroc"),
    ("s_nla", "nla", "rate"), ("s_ol", "ol", "rate"), ("x_ol", "ol", "rate"),
]
X_COL0, COL_W, GAP = 3.56, 0.89, 0.18


def col_x(i: int) -> float:
    return X_COL0 + i * COL_W + (GAP if i >= 3 else 0.0)


def fmt_rate(r: float) -> str:
    if r == 0:
        return "0%"
    if r < 0.005:
        return f"{100 * r:.1f}%"
    return f"{100 * r:.0f}%"


def compute():
    vals, srcs = {}, set()
    for _, rows in GROUPS:
        for f, _ in rows:
            v = {}
            for key, r in (("p_nla", "nla"), ("p_ol", "ol"), ("p_jl", "jlens")):
                p = g.prompted(f, r, n_boot=200)
                v[key] = p["reported"]
                v[key + "_n"] = p["n"]
                srcs.update(p["sources"])
            sn, so = g.states_rate(f, "nla", n_boot=200), g.states_rate(f, "ol42", n_boot=200)
            srcs.update([sn["source"], so["source"]])
            v.update(s_nla=sn["rate"], s_ol=so["rate"], x_ol=so["exhibit_rate"], n_nla=sn["n"], n_ol=so["n"],
                     k_ol=so["k"], items=so["n_items"])
            vals[f] = v
    return vals, srcs


def cell(ax, x, y, h, val, kind, ck):
    if kind == "auroc":
        txt = f"{val:.2f}"
        tint = round(val, 2) >= READABLE
        bold = tint
        color = GREY if round(val, 2) < NEAR_CHANCE else C["ink"]
    else:
        txt = fmt_rate(val)
        tint = val > 0
        bold = val >= 0.01  # a single readout of 320 (0.3%) is tinted but not bold
        color = C["ink"] if val > 0 else GREY
    if tint:
        ax.add_patch(Rectangle((x + 0.05, y - h / 2 + 0.018), COL_W - 0.10, h - 0.036, facecolor=to_rgba(C[ck], TINT),
                               edgecolor="none", zorder=1))
    ax.text(x + COL_W / 2, y, txt, ha="center", va="center", fontsize=FS, color=color, zorder=2,
            fontweight="bold" if bold else "normal")


def header(ax, H):
    """Three header lines: question (bold) / what is measured / reader names (+ verb for unprompted)."""
    for i0, i1, lab, sub in ((0, 2, "Asked", "prompted AUROC"), (3, 5, "Unprompted", "% of 320 free readouts")):
        x0, x1 = col_x(i0) + 0.04, col_x(i1) + COL_W - 0.04
        xc = (x0 + x1) / 2
        ax.text(xc, H - 0.13, lab, ha="center", va="center", fontsize=FS_HEAD, fontweight="bold", color=C["ink"])
        ax.text(xc, H - 0.37, sub, ha="center", va="center", fontsize=FS, color=C["muted"])
        ax.plot([x0, x1], [H - 0.53, H - 0.53], color=C["muted"], lw=1.0)
    ya, yb = H - 0.69, H - 0.92
    kw = dict(ha="center", va="center", fontsize=FS, fontweight="bold", linespacing=1.0)
    for i, (lab, ck) in enumerate((("NLA", "nla"), ("Oracle\nLens", "ol"), ("J-lens", "jl"))):
        ax.text(col_x(i) + COL_W / 2, (ya + yb) / 2, lab, color=C[ck], **kw)
    ax.text(col_x(3) + COL_W / 2, ya, "NLA", color=C["nla"], **kw)
    xo0, xo1 = col_x(4) + 0.08, col_x(5) + COL_W - 0.08
    ax.text((xo0 + xo1) / 2, ya, "Oracle Lens", color=C["ol"], **kw)
    for i, verb in ((3, "states"), (4, "states"), (5, "exhibits")):
        ax.text(col_x(i) + COL_W / 2, yb, verb, color=C["ink"], **kw)


def draw(vals):
    style.slide_style()
    W, H = style.SLIDE_NET
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    n_rows = sum(len(r) for _, r in GROUPS)
    top_body = H - 1.07
    group_gap = 0.07
    rh = (top_body - 0.01 - group_gap * (len(GROUPS) - 1)) / n_rows
    header(ax, H)
    right = col_x(5) + COL_W
    ax.plot([X_FAM, right], [top_body + 0.02, top_body + 0.02], color=C["ink"], lw=1.2)
    y = top_body
    for gi, (fam, rows) in enumerate(GROUPS):
        if gi:
            y -= group_gap
            ax.plot([X_FAM, right], [y + group_gap / 2, y + group_gap / 2], color=C["line"], lw=1.1)
        g_top = y
        for f, lab in rows:
            yc = y - rh / 2
            ax.text(X_ROW, yc, lab, ha="left", va="center", fontsize=FS, color=C["ink"])
            for i, (key, ck, kind) in enumerate(COLS):
                cell(ax, col_x(i), yc, rh, vals[f][key], kind, ck)
            y -= rh
        ax.text(X_FAM + 0.02, g_top - rh / 2, fam, ha="left", va="top" if "\n" in fam else "center", fontsize=FS,
                color=C["muted"], style="italic", linespacing=1.0)
    return fig


def main() -> None:
    vals, srcs = compute()
    fig = draw(vals)

    # ---- checks against the existing backup manifest (must match exactly) and exp 12's combined.json
    man = json.loads((style.MANIFEST / "bk_exp12_recall.json").read_text())
    rec = {c["quantity"]: c["computed"] for c in man["checks"]}
    ol_states = sum(v["k_ol"] for v in vals.values())
    ol_n = sum(v["n_ol"] for v in vals.values())
    ge = sum(v["p_ol"] >= v["p_nla"] for v in vals.values())
    mp = "figures/manifest/bk_exp12_recall.json"
    checks = [
        g.check("rows with Oracle Lens >= NLA prompted (unrounded)", ge, rec["rows with Oracle Lens >= NLA prompted (unrounded)"], mp, tol=0),
        g.check("Oracle Lens 'states' verdicts at layer 42", f"{ol_states}/{ol_n}", rec["Oracle Lens 'states' verdicts at layer 42"], mp),
        g.check("list item index NLA stated", round(vals["bk_listidx"]["s_nla"], 2), rec["list item index NLA stated"], mp, tol=0),
        g.check("code language Oracle Lens exhibits", round(vals["bk_code"]["x_ol"], 2), rec["code language Oracle Lens exhibits"], mp, tol=0),
        g.check("caps Oracle Lens exhibits", round(vals["bk_caps"]["x_ol"], 2), rec["caps Oracle Lens exhibits"], mp, tol=0),
    ]
    for f in ("nj1_col", "bk_wordpos", "bk_indent", "bk_quote"):
        checks.append(g.check(f"{f} NLA prompted", round(vals[f]["p_nla"], 2), rec[f"{f} NLA prompted"], mp, tol=0))
    comb_p = g.E12 / "results" / "combined.json"
    comb = {r["feature"]: r for r in json.loads(comb_p.read_text())}
    cmap = {"p_nla": "nla_auroc", "p_ol": "ol_auroc", "p_jl": "jlens_auroc", "s_nla": "nla_stated", "s_ol": "ol_stated"}
    n_ok = n_all = 0
    for f, v in vals.items():
        for k, ck in cmap.items():
            n_all += 1
            ok = abs(v[k] - comb[f][ck]) <= 0.0051
            n_ok += ok
            if not ok:
                checks.append(g.check(f"{f} {k}", round(v[k], 4), comb[f][ck], g.rel(comb_p)))
    checks.append(g.check("cells agreeing with exp 12 combined.json (AUROC x3, NLA/Oracle Lens states)", f"{n_ok}/{n_all}",
                          f"{n_all}/{n_all}", g.rel(comb_p)))
    # exhibits: combined.json 'ol_exhibited' vs our 'exhibited the true value'
    ex_ok = sum(abs(v["x_ol"] - comb[f]["ol_exhibited"]) <= 0.0051 for f, v in vals.items())
    checks.append(g.check("Oracle Lens exhibits rate agreeing with combined.json", f"{ex_ok}/14", "14/14", g.rel(comb_p)))
    srcs.add(g.rel(comb_p))

    fv = list(vals.values())
    hi_nla = sum(round(v["p_nla"], 2) >= READABLE for v in fv)
    hi_ol = sum(round(v["p_ol"], 2) >= READABLE for v in fv)
    hi_jl = sum(round(v["p_jl"], 2) >= READABLE for v in fv)
    exhib = [f for f, v in vals.items() if v["x_ol"] > 0]
    checks.append(g.check("concepts Oracle Lens exhibits (> 0%)", sorted(exhib), sorted(["ij2_lang", "bk_code", "bk_caps", "nj2_tense"]),
                          "exp 12 RESULTS point 1 / bk_exp12_recall title"))
    n_items = {v["p_nla_n"] for v in fv} | {v["p_ol_n"] for v in fv}
    title = (f"Asked, NLA and Oracle Lens detect most concepts (AUROC 0.80 or more on {hi_nla} and {hi_ol} of 14, J-lens {hi_jl}); "
             f"unprompted, Oracle Lens states {ol_states} in {ol_n:,} readouts and exhibits only language, code, caps and tense")
    save(fig, "s3_concept_table", title=title,
         caption=(f"Exp 12, layer 42, 14 concepts. Asked: prompted AUROC (0.5 = chance), the better of the mean of 3 mined "
                  f"phrasings and the fixed label, n = {', '.join(map(str, sorted(n_items)))} items per cell. Unprompted: share of 320 free readouts "
                  "per concept (40 items x 8 samples) the judge (Gemini 3.8 Flash) says state the value, or for Oracle Lens "
                  "whose own text exhibits it. Shading: reader-colour tint = AUROC 0.80 or more / any readout > 0%; grey = "
                  "AUROC below 0.60 or 0%. No intervals in cells (AUROC 95% bootstrap clustered by source; rates "
                  "item-clustered bootstrap or Wilson at 0): see backup b21 (bk_exp12_recall) for them."),
         section="S3", role="main", sources=sorted(srcs), checks=checks,
         notes=("No probe readability or J-enrichment column: exp 12's raw files hold neither per concept, so it is omitted. "
                f"Oracle Lens's {ol_states} 'states' verdict is on French or German ({fmt_rate(vals['ij2_lang']['s_ol'])} of its 320). "
                "Exhibits = the readout's own text shows the true value (e.g. it is written in German), not a claim about it. "
                "Values match bk_exp12_recall.json exactly (checks). Text is 13 pt at 9.4 in (a table may go to 13 pt)."))


if __name__ == "__main__":
    main()
