"""bk_exp13_new_rows: all exp 13 rows, prompted AUROC per reader against prompt length alone, and stated rates."""
from __future__ import annotations

import json

import g2_common as g
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from style import C, auroc, save

ROWS = [("tb_tablecol", "table column (first vs 4th-6th)"), ("tb_paren", "inside vs outside a parenthesis"),
        ("tb_sentcount", "sentences into the paragraph"), ("tb_codeline", "line in a function body"),
        ("tb_wordsleft", "words left under a stated limit"), ("nj1_col_fresh", "line position, fresh items"),
        ("fmt_qa", "Q&A vs prose layout"), ("fmt_listdepth", "list nesting depth")]
CK = {"nla": "nla", "ol": "ol", "jlens": "jl"}


def main() -> None:
    items = {it["id"]: it for it in json.loads((g.E13 / "results" / "items_base.json").read_text())}
    ypos = np.arange(len(ROWS))[::-1].astype(float)
    ypos[6:] -= 0.6
    P, S, L, srcs = {}, {}, {}, {g.rel(g.E13 / "results" / "items_base.json")}
    for f, _ in ROWS:
        for r in ("nla", "ol", "jlens"):
            P[(f, r)] = g.prompted(f, r, n_boot=1000)
            srcs.update(P[(f, r)]["sources"])
        sc = json.loads((g.E13 / "results" / f"score_nla_{f}.json").read_text())["scores"]["nla@42"]
        ln = np.array([len(items[i]["ids"]) for i in sc["ids"]], float)
        a = auroc(ln, np.array(sc["y"]))
        L[f] = (a, max(a, 1 - a))
        if f != "nj1_col_fresh":
            for r in ("nla", "ol42"):
                S[(f, r)] = g.states_rate(f, r, n_boot=1000)
                srcs.add(S[(f, r)]["source"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 6.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.06})
    off = {"nla": 0.2, "ol": 0.0, "jlens": -0.2}
    for (f, lab), y0 in zip(ROWS, ypos):
        for r, name in (("nla", "NLA"), ("ol", "Oracle Lens"), ("jlens", "J-lens")):
            p = P[(f, r)]
            ax.plot(p["ci"], [y0 + off[r]] * 2, color=C[CK[r]], lw=1.8, alpha=0.5)
            ax.plot(p["reported"], y0 + off[r], "o", ms=8, color=C[CK[r]], mec="white", mew=0.8,
                    label=name if f == "tb_tablecol" else None)
        ax.plot(L[f][1], y0, "|", ms=22, mew=3, color=C["text"], label="prompt length alone" if f == "tb_tablecol" else None)
    ax.axvline(0.5, color=C["line"], ls="--", lw=1.4, zorder=0)
    ax.axvline(0.8, color=C["line"], ls=":", lw=1.2, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=12.5)
    ax.set_xlim(0.35, 1.03)
    ax.set_ylim(ypos[-1] - 0.6, ypos[0] + 0.9)
    ax.text(0.5, ypos[0] + 0.55, "chance", ha="center", fontsize=11, color=C["muted"])
    ax.text(0.8, ypos[0] + 0.55, "0.80", ha="center", fontsize=11, color=C["muted"])
    ax.set_xlabel("Prompted AUROC (0.5 = chance)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, fontsize=11.5, handletextpad=0.2, columnspacing=1.0)
    for a in (ax, bx):
        a.axhline((ypos[5] + ypos[6]) / 2, color=C["line"], lw=1)
    for (f, lab), y0 in zip(ROWS, ypos):
        if f == "nj1_col_fresh":
            bx.text(0.01, y0, "not judged", va="center", fontsize=11.5, color=C["muted"])
            continue
        s, o = S[(f, "nla")], S[(f, "ol42")]
        bx.barh(y0 + 0.17, s["rate"], height=0.32, color=C["nla"])
        if s["k"]:
            bx.plot(s["ci"], [y0 + 0.17] * 2, color=C["ink"], lw=1)
            bx.text(max(s["ci"][1], s["rate"]) + 0.01, y0 + 0.17, f"{100 * s['rate']:.0f}%", va="center", fontsize=11, color=C["nla"])
        bx.barh(y0 - 0.17, o["rate"], height=0.32, color=C["ol"])
        bx.text(0.01, y0 - 0.17, f"{o['k']}/{o['n']}", va="center", fontsize=10, color=C["ol"])
    bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_ylim(ax.get_ylim())
    bx.set_xlim(0, 0.75)
    bx.set_xticks([0, 0.25, 0.5, 0.75])
    bx.set_xticklabels(["0", "25", "50", "75"])
    bx.set_xlabel("Free readouts that state the value (%)")
    bx.legend(handles=[Patch(color=C["nla"], label="NLA"), Patch(color=C["ol"], label="Oracle Lens")], loc="lower right",
              fontsize=11.5)
    rec = {("nj1_col_fresh", "nla"): 0.51, ("nj1_col_fresh", "ol"): 0.49, ("nj1_col_fresh", "jlens"): 0.53,
           ("tb_sentcount", "nla"): 0.99, ("tb_codeline", "nla"): 0.99, ("tb_wordsleft", "nla"): 0.58, ("fmt_qa", "nla"): 0.79}
    checks = [g.check(f"{f} {r} prompted", round(P[(f, r)]["reported"], 2), v, "exp 13 RESULTS prompted table") for (f, r), v in rec.items()]
    checks.append(g.check("line position fresh NLA mean of 3 (RESULTS text '0.48')", round(P[("nj1_col_fresh", "nla")]["mean3"], 2), 0.48,
                          "exp 13 RESULTS text"))
    checks.append({"quantity": "line position fresh NLA: table vs text", "computed": f"mean3 {P[('nj1_col_fresh','nla')]['mean3']:.3f}, fixed {P[('nj1_col_fresh','nla')]['fixed']:.3f}",
                   "record": "RESULTS table 0.51 and talk script 0.51; RESULTS text 'NLA 0.48'", "where": "exp 13 RESULTS", "match": False})
    lrec = {"tb_tablecol": 0.32, "tb_paren": 0.29, "tb_sentcount": 0.00, "tb_codeline": 0.00, "fmt_listdepth": 0.42, "fmt_qa": 0.86}
    checks += [g.check(f"{f} length-alone AUROC", round(L[f][0], 2), v, "exp 13 RESULTS prompted table", tol=0.011) for f, v in lrec.items()]
    checks.append(g.check("Oracle Lens states, all exp 13 rows", sum(S[k]["k"] for k in S if k[1] == "ol42"), 0, "exp 13 RESULTS (0 of 2,240)"))
    save(fig, "bk_exp13_new_rows",
         title="Table column and parenthesis are the only new rows the NLA both reads beyond length and states",
         caption=("Exp 13, layer 42, 160 held-out items per row. Left: prompted AUROC per reader (95% bootstrap CI clustered by passage); "
                  "the ochre tick is how well prompt length alone separates the classes (either direction), so a reader at or below it may "
                  "be reading length: sentence count and code-line number are perfectly length-confounded by construction. Right: share of "
                  "320 free readouts judged to state the value; Oracle Lens 0 of 2,240. Line position on fresh items is at chance for all "
                  "readers (NLA 0.48 mean-of-3, 0.51 fixed label)."),
         section="backup:exp13", role="backup", sources=sorted(srcs), checks=checks,
         notes="Length = token count of the item prefix (items_base.json 'ids'), AUROC vs the class label, reported as max(a, 1-a).")


if __name__ == "__main__":
    main()
