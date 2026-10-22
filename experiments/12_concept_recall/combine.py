"""Combine exp 07 prompted AUROCs with exp 12 spontaneous recall into one table (md, json, png)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import NAMES  # noqa: E402

HERE = Path(__file__).resolve().parent
SEL = HERE.parents[1] / "experiments" / "07_template_mining" / "results" / "selection_table.json"
ORDER = ["nj1_col", "nj1_col_break", "bk_wordpos", "bk_listidx", "bk_indent", "bk_quote", "nj3_predpos",
         "agr_gender", "agr_number", "nj2_tense", "bk_caps", "ij_listnum", "ij2_lang", "bk_code"]
GROUP = {f: "text-buffer structure" for f in ORDER[:6]}
GROUP.update({f: "upcoming-word syntax" for f in ORDER[6:9]})
GROUP.update({f: "controls (in-workspace)" for f in ORDER[9:]})


def prompted() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for feat, reader, _sel, mean3, fixed, *_ in json.loads(SEL.read_text()):
        vals = [x for x in (mean3, fixed) if x is not None]
        out.setdefault(feat, {})[reader] = max(vals) if vals else float("nan")
    return out


def main() -> None:
    p = prompted()
    s = json.loads((HERE / "results" / "recall_table.json").read_text())
    rows = []
    for f in ORDER:
        nla, ol = s[f]["nla"], s[f]["ol42"]
        rows.append({"feature": f, "name": NAMES[f], "group": GROUP[f],
                     "nla_auroc": p[f]["nla"], "ol_auroc": p[f]["ol"], "jlens_auroc": p[f]["jlens"],
                     "nla_stated": nla["stated_rate"], "nla_any8": nla["any_correct_per_item"],
                     "ol_stated": ol["stated_rate"], "ol_exhibited": ol["exhibited_correct_rate"]})
    (HERE / "results" / "combined.json").write_text(json.dumps(rows, indent=1))
    hdr = ("| concept type | group | NLA prompted | OLens prompted | J-lens prompted | "
           "NLA states it | NLA any-of-8 | OLens states it | OLens exhibits it |\n|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['name']} | {r['group']} | {r['nla_auroc']:.2f} | {r['ol_auroc']:.2f} | {r['jlens_auroc']:.2f} | "
        f"{100*r['nla_stated']:.0f}% | {100*r['nla_any8']:.0f}% | {100*r['ol_stated']:.0f}% | "
        f"{100*r['ol_exhibited']:.0f}% |\n" for r in rows)
    (HERE / "results" / "combined.md").write_text(hdr + body)
    print(hdr + body)

    cols = ["NLA\nprompted", "OLens\nprompted", "J-lens\nprompted", "NLA\nstates it", "NLA\nany of 8",
            "OLens\nstates it", "OLens\nexhibits it"]
    cell = [[f"{r['nla_auroc']:.2f}", f"{r['ol_auroc']:.2f}", f"{r['jlens_auroc']:.2f}",
             f"{100*r['nla_stated']:.0f}%", f"{100*r['nla_any8']:.0f}%", f"{100*r['ol_stated']:.0f}%",
             f"{100*r['ol_exhibited']:.0f}%"] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.axis("off")
    t = ax.table(cellText=cell, rowLabels=[f"{r['name']}  ({r['group'].split(' (')[0]})" for r in rows],
                 colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
    for (i, j), c in t.get_celld().items():
        if i == 0 or j < 0:
            c.set_text_props(weight="bold"); c.set_facecolor("#EEF0EC")
        elif j < 3:
            v = float(cell[i-1][j]); c.set_facecolor(plt.cm.Oranges(min(1, max(0, (v-0.5)/0.5))*0.6+0.0) if v > 0.5 else "#FFFFFF")
        else:
            v = float(cell[i-1][j].rstrip("%"))/100; c.set_facecolor(plt.cm.Blues(v*0.6))
    ax.set_title("Qwen3.6-27B, layer 42, one activation per item.  Prompted = AUROC between two frozen phrasings, "
                 "160 items, 0.5 = chance.\nStates it = % of free readouts (40 items x 8 samples) that explicitly state "
                 "the property.  Exhibits it = the readout's own text shows the true value.",
                 fontsize=9, loc="center")
    fig.tight_layout(); fig.savefig(HERE / "results" / "combined.png", dpi=170)
    print("png written")


if __name__ == "__main__":
    main()
