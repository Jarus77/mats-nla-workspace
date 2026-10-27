"""Exp 27 analysis (laptop): exp 23's analysis on the 2,000-point bank, with exp 23's numbers side by side and the
judge table merged in when results/judge27.json exists.
  naming (strict): share of readouts naming the state when it holds vs when it does not (lib23.names_state), with
          Wilson intervals; the loose any-vocabulary count is kept for the record only;
  carrying: for each state with both classes present (>= 20 each) in a kind, a bag-of-tokens classifier on the readout
          text (grouped by file) and a probe on the layer-42 activation (readable), both AUROC; the all-kinds string
          row is confounded with document kind (JSON points are nearly all in-string), the code-only row is the honest one.
Reads results/items_base.json, results/ol27.json, results/nla27.json, results/vol/acts27.pt,
experiments/23_random_points/results/table23.json, [results/judge27.json] -> results/table27.md, table27.json."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "19_natural_bank"))
sys.path.insert(0, str(REPO / "experiments" / "23_random_points"))
from analyze23 import MIN_EACH, STATES  # noqa: E402
from lib23 import names_structure, naming_by_state  # noqa: E402
from lib27 import render_carrying_table, render_judge_table, render_strict_table  # noqa: E402

R = HERE / "results"
TABLE23 = REPO / "experiments" / "23_random_points" / "results" / "table23.json"


def naming_tables(items: list[dict], ol: dict, nla: dict) -> tuple[dict, dict]:
    kinds = sorted({it["meta"]["kind"] for it in items})
    loose = {}
    for kind in kinds:
        ks = [it for it in items if it["meta"]["kind"] == kind]
        loose[kind] = {"n": len(ks), "ol_names_any": float(np.mean([names_structure(ol[it["id"]]) for it in ks])),
                       "nla_names_any": float(np.mean([names_structure(nla[it["id"]]) for it in ks]))}
    labelled = [{"id": it["id"], "meta": {"labels": {st: fn(it["meta"]["labels"][st]) for st, fn in STATES.items()}}} for it in items]
    strict = {st: {name: naming_by_state(d, labelled, st) for name, d in (("ol", ol), ("nla", nla))} for st in STATES}
    return strict, loose


def carrying_table(items: list[dict], ol: dict, nla: dict, acts) -> list[dict]:  # noqa: ANN001
    from transformers import AutoTokenizer

    from lib19 import bag_of_tokens
    from mats.probes import cv_auroc

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    enc = lambda s: tok(s or "", add_special_tokens=False)["input_ids"]  # noqa: E731
    kinds = sorted({it["meta"]["kind"] for it in items})
    rows = []
    for kind, members in [(k, (k,)) for k in kinds] + [("code (py+js)", ("py", "js")), ("all", tuple(kinds))]:
        idx = [i for i, it in enumerate(items) if it["meta"]["kind"] in members]
        for st, fn in STATES.items():
            y = np.array([fn(items[i]["meta"]["labels"][st]) for i in idx], int)
            c = Counter(y.tolist())
            if c.get(0, 0) < MIN_EACH or c.get(1, 0) < MIN_EACH:
                continue
            g = np.array([items[i]["cluster"] for i in idx])
            row = {"kind": kind, "state": st, "n_pos": int(c[1]), "n_neg": int(c[0]),
                   "readable": cv_auroc(acts["acts"][idx].float(), y, g, k=5)["auroc"]}
            for name, d in (("ol", ol), ("nla", nla)):
                Xt = bag_of_tokens([enc(d[items[i]["id"]]) for i in idx], window=10**6, min_df=3)
                row[f"{name}_text_bag"] = cv_auroc(Xt, y, g, k=5, lam=1.0)["auroc"]
            rows.append(row)
            print(row, flush=True)
    return rows


def render(strict: dict, loose: dict, carrying: list[dict], prev: dict | None, judge: dict | None) -> str:
    md = render_strict_table(strict, prev["naming_strict"] if prev else None)
    md += "\n## Any structural vocabulary at all (loose regex; over-counts content words such as \"open\", \"math\", \"level\")\n\n"
    md += "| kind | n | Oracle Lens | NLA |\n|---|---:|---:|---:|\n"
    md += "".join(f"| {k} | {v['n']} | {v['ol_names_any']:.2f} | {v['nla_names_any']:.2f} |\n" for k, v in loose.items())
    md += "\n" + render_carrying_table(carrying, prev["carrying"] if prev else None)
    if judge:
        md += "\n" + render_judge_table(judge["rates"])
        md += f"\n(judge: {judge.get('model')}, T={judge.get('temperature')}; {judge['n_judged']} of {judge['n_rows']} rows judged, {judge['n_unparsed']} unparsed)\n"
    else:
        md += "\n## Blind judge on the in-string readouts\n\n(pending: run judge27.py, then analyze27.py again)\n"
    return md


def main() -> None:
    import torch

    items = json.load(open(R / "items_base.json"))
    ol = json.load(open(R / "ol27.json"))["ol"]
    nla = json.load(open(R / "nla27.json"))["nla"]
    acts = torch.load(R / "vol" / "acts27.pt", weights_only=False)
    assert acts["ids"] == [it["id"] for it in items]
    prev = json.load(open(TABLE23)) if TABLE23.exists() else None
    judge = json.load(open(R / "judge27.json")) if (R / "judge27.json").exists() else None
    strict, loose = naming_tables(items, ol, nla)
    carrying = carrying_table(items, ol, nla, acts)
    md = render(strict, loose, carrying, prev, judge)
    (R / "table27.md").write_text(md)
    (R / "table27.json").write_text(json.dumps({"n_items": len(items), "naming_strict": strict, "naming_loose": loose, "carrying": carrying,
                                                "judge": judge["rates"] if judge else None, "exp23": prev}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
