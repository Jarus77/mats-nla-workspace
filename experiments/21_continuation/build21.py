"""Exp 21 (text only, no GPU): for each exp 19 row, recover the true text after each read point from the same local
source files (located by the bank's cluster name = basename + sha1 of the file text), take the next 32 tokens, and
ask a bag-of-tokens classifier (grouped by file) whether the state changes the continuation. For exp 13's table-column
row, whose sources were synthetic and whose items store only the prefix, the continuation is regenerated from the
generator's cell pools given the item's column and width (labelled synthetic). Joins the result with exp 20's
Oracle Lens readout AUROC per row and reports the rank correlation.

    .venv/bin/python build21.py [--home ~] [--arxiv <dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "19_natural_bank"))
sys.path.insert(0, str(REPO / "experiments" / "13_new_rows"))
import build19 as B19  # noqa: E402
from lib21 import N_NEXT, continuation_auroc, locate  # noqa: E402

E19 = REPO / "experiments" / "19_natural_bank" / "results" / "items_base.json"
E13 = REPO / "experiments" / "13_new_rows" / "results" / "items_base.json"
E20 = REPO / "experiments" / "20_ar_structure" / "results"
MARKED = {"py_string": "inside", "js_string": "inside", "latex_math": "inside", "py_arg": "third", "json_depth": "deep",
          "html_depth": "deep", "md_heading": "deep", "md_list": "nested", "tb_tablecol": "first"}


def source_index(home: Path, arxiv: Path | None, exts: set[str]) -> dict[str, str]:
    """cluster name -> file text, for every candidate source file of the needed extensions."""
    out = {}
    for ext in exts:
        for p in B19.find_files(home, ext, arxiv):
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if B19.MIN_CHARS <= len(t) <= B19.MAX_CHARS:
                out[f"{p.name}_{hashlib.sha1(t.encode()).hexdigest()[:12]}"] = t
    return out


def tablecol_continuation(it: dict, rng: random.Random) -> str:
    import items_build as IB13
    m = it["meta"]
    heads = it["text"].splitlines()[0].strip("| ").split(" | ")
    rest = [rng.choice(IB13._COLS[h]) for h in heads[m["col"]:]]
    return (" | " + " | ".join(rest) if rest else "") + " |\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(Path.home()))
    ap.add_argument("--arxiv", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    items19 = json.load(open(E19))
    rows = sorted({it["feature"] for it in items19})
    src = source_index(Path(a.home), Path(a.arxiv) if a.arxiv else None, {B19.EXT[r] for r in rows})
    res = {}
    for row in rows + ["tb_tablecol"]:
        if row == "tb_tablecol":
            its = [it for it in json.load(open(E13)) if it["feature"] == row]
            rng = random.Random(21)
            conts = [tablecol_continuation(it, rng) for it in its]
            found = list(range(len(its)))
        else:
            its = [it for it in items19 if it["feature"] == row]
            conts, found = [], []
            for i, it in enumerate(its):
                t = src.get(it["cluster"])
                if t is None:
                    continue
                off = locate(t, it["text"])
                if off is None:
                    continue
                conts.append(t[off: off + 600])
                found.append(i)
        ids = [enc(c)[:N_NEXT] for c in conts]
        y = np.array([its[i]["value"] == MARKED[row] for i in found], int)
        g = np.array([its[i]["cluster"] for i in found])
        auroc = continuation_auroc(ids, y, g) if len(found) >= 40 and len(set(y)) == 2 else None
        ol = None
        p = E20 / f"recon20_{row}.json"
        if p.exists():
            d = json.loads(p.read_text())
            ol = {"ol_readout": d["auroc"]["ol_readout"], "nla_readout": d["auroc"]["nla_readout"], "model_span_32": d["auroc"]["model_span_32"]}
        res[row] = {"n_found": len(found), "n_items": len(its), "continuation_auroc": auroc, "exp20": ol,
                    "synthetic_continuation": row == "tb_tablecol"}
        print(row.ljust(11), "found", f"{len(found)}/{len(its)}", "cont", None if auroc is None else round(auroc, 3), "ol", None if ol is None else round(ol["ol_readout"], 2), flush=True)
    pairs = [(v["continuation_auroc"], v["exp20"]["ol_readout"]) for v in res.values() if v["continuation_auroc"] is not None and v["exp20"]]
    from scipy.stats import spearmanr
    rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None
    out = {"rows": res, "spearman_cont_vs_ol_readout": None if rho is None else {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(pairs)}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "table21.json").write_text(json.dumps(out, indent=1))
    print("spearman", out["spearman_cont_vs_ol_readout"])


if __name__ == "__main__":
    main()
