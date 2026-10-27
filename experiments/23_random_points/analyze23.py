"""Exp 23 analysis (laptop): base rates at random read points.
  naming: share of readouts (per reader, per kind) whose text names any structural state (lib23.MENTION_ANY), and an
          LLM-free check that the named state is the true one is left to the judge step if ever needed;
  carrying: for each state with both classes present (>= 20 each) in a kind, a bag-of-tokens classifier on the readout
          text (grouped by file) and a probe on the layer-42 activation (readable), both AUROC.
Reads results/items_base.json, results/ol23.json, results/nla23.json, results/vol/acts23.pt -> results/table23.md, table23.json."""
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
from lib23 import names_structure  # noqa: E402

STATES = {"in_string": lambda v: bool(v), "in_math": lambda v: bool(v), "paren_depth": lambda v: v > 0,
          "brace_depth": lambda v: v >= 2, "list_depth": lambda v: v > 0, "heading_level": lambda v: v > 0}
MIN_EACH = 20


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    from lib19 import bag_of_tokens
    from mats.probes import cv_auroc

    R = HERE / "results"
    items = json.load(open(R / "items_base.json"))
    ol = json.load(open(R / "ol23.json"))["ol"]
    nla = json.load(open(R / "nla23.json"))["nla"]
    acts = torch.load(R / "vol" / "acts23.pt", weights_only=False)
    assert acts["ids"] == [it["id"] for it in items]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    enc = lambda s: tok(s or "", add_special_tokens=False)["input_ids"]  # noqa: E731
    kinds = sorted({it["meta"]["kind"] for it in items})
    naming = {}
    for kind in kinds:
        ks = [it for it in items if it["meta"]["kind"] == kind]
        naming[kind] = {"n": len(ks), "ol_names_any": float(np.mean([names_structure(ol[it["id"]]) for it in ks])),
                        "nla_names_any": float(np.mean([names_structure(nla[it["id"]]) for it in ks]))}
    carrying = []
    groups_of_kinds = [(k, (k,)) for k in kinds] + [("code (py+js)", ("py", "js")), ("all", tuple(kinds))]
    for kind, members in groups_of_kinds:
        idx = [i for i, it in enumerate(items) if it["meta"]["kind"] in members]
        for st, fn in STATES.items():
            y = np.array([fn(items[i]["meta"]["labels"][st]) for i in idx], int)
            c = Counter(y.tolist())
            if c.get(0, 0) < MIN_EACH or c.get(1, 0) < MIN_EACH:
                continue
            g = np.array([items[i]["cluster"] for i in idx])
            X = acts["acts"][idx].float()
            probe = cv_auroc(X, y, g, k=5)["auroc"]
            row = {"kind": kind, "state": st, "n_pos": int(c[1]), "n_neg": int(c[0]), "readable": probe}
            for name, d in (("ol", ol), ("nla", nla)):
                Xt = bag_of_tokens([enc(d[items[i]["id"]]) for i in idx], window=10**6, min_df=3)
                row[f"{name}_text_bag"] = cv_auroc(Xt, y, g, k=5, lam=1.0)["auroc"]
            carrying.append(row)
            print(row, flush=True)
    lines = ["## Naming at random read points (share of readouts naming any structural state)", "",
             "| kind | n | Oracle Lens | NLA |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['n']} | {v['ol_names_any']:.2f} | {v['nla_names_any']:.2f} |" for k, v in naming.items()]
    lines += ["", "## Carrying at random read points (AUROC; readable = probe on the activation; text bag = classifier on the readout alone)", "",
              "| kind | state | n yes / no | readable | Oracle Lens readout | NLA readout |", "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['kind']} | {r['state']} | {r['n_pos']} / {r['n_neg']} | {r['readable']:.2f} | {r['ol_text_bag']:.2f} | {r['nla_text_bag']:.2f} |" for r in carrying]
    (R / "table23.md").write_text("\n".join(lines) + "\n")
    (R / "table23.json").write_text(json.dumps({"naming": naming, "carrying": carrying}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
