"""Extended concept x reader table: the 14 exp 07/12 rows plus the exp 13 rows (md, json, png)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
E12 = HERE.parents[1] / "experiments" / "12_concept_recall"
sys.path.insert(0, str(E12))
from recall_lib import aggregate  # noqa: E402

NAMES = {"tb_sentcount": "sentences into paragraph", "tb_tablecol": "table column", "tb_codeline": "line in function body",
         "tb_paren": "inside a parenthesis", "tb_wordsleft": "words left under a stated limit",
         "fmt_qa": "Q&A vs prose layout", "fmt_listdepth": "list nesting depth", "nj1_col_fresh": "line position, fresh items"}
GROUP = {"tb_sentcount": "text-buffer structure", "tb_tablecol": "text-buffer structure", "tb_codeline": "text-buffer structure",
         "tb_paren": "text-buffer structure", "tb_wordsleft": "text-buffer structure", "nj1_col_fresh": "text-buffer structure",
         "fmt_qa": "formatting", "fmt_listdepth": "formatting"}
ORDER = ["nj1_col_fresh", "tb_sentcount", "tb_tablecol", "tb_codeline", "tb_paren", "tb_wordsleft", "fmt_qa", "fmt_listdepth"]


def prompted13() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for feat, reader, _sel, mean3, fixed, *_ in json.loads((HERE / "results" / "selection_table.json").read_text()):
        vals = [x for x in (mean3, fixed) if x is not None]
        out.setdefault(feat, {})[reader] = max(vals) if vals else float("nan")
    return out


def spontaneous13(feat: str, reader: str) -> dict | None:
    p = HERE / "results" / "judged" / f"{feat}_{reader}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return aggregate(d["judged"], d["values"])


def rows13() -> list[dict]:
    p = prompted13()
    out = []
    for f in ORDER:
        n, o = spontaneous13(f, "nla"), spontaneous13(f, "ol42")
        out.append({"feature": f, "name": NAMES[f], "group": GROUP[f], "new": True,
                    "nla_auroc": p.get(f, {}).get("nla", float("nan")), "ol_auroc": p.get(f, {}).get("ol", float("nan")),
                    "jlens_auroc": p.get(f, {}).get("jlens", float("nan")),
                    "nla_stated": n["stated_rate"] if n else float("nan"), "nla_any8": n["any_correct_per_item"] if n else float("nan"),
                    "ol_stated": o["stated_rate"] if o else float("nan"), "ol_exhibited": o["exhibited_correct_rate"] if o else float("nan")})
    return out


def fmt(x: float, pct: bool) -> str:
    if x != x:
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.2f}"


def main() -> None:
    old = [{**r, "new": False} for r in json.loads((E12 / "results" / "combined.json").read_text())]
    new = rows13()
    rows = new + old
    (HERE / "results" / "combined_all.json").write_text(json.dumps(rows, indent=1))
    cols = [("nla_auroc", False), ("ol_auroc", False), ("jlens_auroc", False), ("nla_stated", True), ("nla_any8", True),
            ("ol_stated", True), ("ol_exhibited", True)]
    hdr = ("| concept type | group | NLA prompted | OLens prompted | J-lens prompted | NLA states it | NLA any-of-8 | "
           "OLens states it | OLens exhibits it |\n|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {'**' if r['new'] else ''}{r['name']}{'**' if r['new'] else ''} | {r['group']} | "
                   + " | ".join(fmt(r[c], p) for c, p in cols) + " |\n" for r in rows)
    (HERE / "results" / "combined_all.md").write_text(hdr + body)
    print(hdr + body)
    labels = ["NLA\nprompted", "OLens\nprompted", "J-lens\nprompted", "NLA\nstates it", "NLA\nany of 8", "OLens\nstates it", "OLens\nexhibits it"]
    cell = [[fmt(r[c], p) for c, p in cols] for r in rows]
    fig, ax = plt.subplots(figsize=(13.5, 10.5))
    ax.axis("off")
    t = ax.table(cellText=cell, rowLabels=[("* " if r["new"] else "") + f"{r['name']}  ({r['group'].split(' (')[0]})" for r in rows],
                 colLabels=labels, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.45)
    for (i, j), c in t.get_celld().items():
        if i == 0 or j < 0:
            c.set_text_props(weight="bold"); c.set_facecolor("#EEF0EC")
            continue
        s = cell[i - 1][j]
        if s == "–":
            continue
        if j < 3:
            v = float(s); c.set_facecolor(plt.cm.Oranges(max(0.0, (v - 0.5) / 0.5) * 0.6) if v > 0.5 else "#FFFFFF")
        else:
            c.set_facecolor(plt.cm.Blues(float(s.rstrip("%")) / 100 * 0.6))
    ax.set_title("Qwen3.6-27B, layer 42, one activation per item.  * = new rows (exp 13).  Prompted = AUROC between two "
                 "frozen phrasings, 160 items, 0.5 = chance.\nStates it = % of free readouts (40 items x 8 samples) that "
                 "explicitly state the property.  Exhibits it = the readout's own text shows the true value.", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "results" / "combined_all.png", dpi=170)
    print("png written")


if __name__ == "__main__":
    main()
