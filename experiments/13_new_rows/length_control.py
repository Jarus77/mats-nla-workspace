"""Length confound check for the exp 13 rows: AUROC of token count alone, and the readers' mean-of-3
template AUROC restricted to items whose token count lies in the range shared by both values."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = ["tb_sentcount", "tb_tablecol", "tb_codeline", "tb_paren", "tb_wordsleft", "fmt_qa", "fmt_listdepth"]


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    items = json.loads((HERE / "results" / "items_base.json").read_text())
    length = {it["id"]: len(it["ids"]) for it in items}
    frozen = json.loads((HERE / "templates_frozen.json").read_text())["templates"]
    out = {}
    for f in ROWS:
        rec = {}
        for reader, key in (("nla", "nla@42"), ("ol", "ol@42")):
            sc = json.loads((HERE / "results" / f"score_{reader}_{f}.json").read_text())["scores"][key]
            y = np.array(sc["y"]); ln = np.array([length[i] for i in sc["ids"]], float)
            lo = np.mean([np.array(sc["lo"][t["id"]]) for t in frozen[reader][f]["selected"]], axis=0)
            lo_, hi_ = max(ln[y == 1].min(), ln[y == 0].min()), min(ln[y == 1].max(), ln[y == 0].max())
            m = (ln >= lo_) & (ln <= hi_)
            ok = m.sum() > 10 and len(set(y[m])) == 2
            rec[reader] = {"auroc_full": round(auroc(lo, y), 3), "n_overlap": int(m.sum()),
                           "auroc_overlap": round(auroc(lo[m], y[m]), 3) if ok else None}
            rec["length_alone_full"] = round(auroc(ln, y), 3)
            rec["length_alone_overlap"] = round(auroc(ln[m], y[m]), 3) if ok else None
        out[f] = rec
        print(f, json.dumps(rec))
    (HERE / "results" / "length_control.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
