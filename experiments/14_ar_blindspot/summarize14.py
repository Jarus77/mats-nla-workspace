"""Local: results/analysis14_*.json -> results/summary.md (+ summary.json). Run with .venv python."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyze14 import table_rows  # noqa: E402
from spec14 import ROWS  # noqa: E402

NAMES = {"tb_tablecol": "table column", "tb_paren": "inside a parenthesis", "ij2_lang": "language (FR/DE)",
         "bk_listidx": "bullet index"}
INPUTS = {"span32": "true text, last 32 tokens", "span128": "true text, last 128 tokens",
          "ol_bullets": "OLens bullets", "nla_expl": "NLA explanation"}


def main() -> None:
    res = {f: json.loads(p.read_text()) for f in ROWS if (p := HERE / "results" / f"analysis14_{f}.json").exists()}
    rows = [r for f in res for r in table_rows(res[f])]
    hdr = ("| concept | ruler | layer | input | FVE | ret@1 | probe on true | probe on recon | probe on error | "
           "probe retrained on recon | gap kept |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {NAMES[r['feature']]} | {r['ruler']} | {r['layer']} | {INPUTS[r['input']]} | {r['fve']:.2f} | "
                   f"{r['ret1']:.2f} | {r['probe_true']:.2f} | {r['probe_recon']:.2f} | {r['probe_err']:.2f} | "
                   f"{r['probe_retrained']:.2f} | {r['gap_ratio']:.2f} |\n" for r in rows)
    st = "| concept | ruler | layer | statement pair | shift along concept (1 = true gap) | cosine |\n|---|---|---|---|---|---|\n"
    for f, d in res.items():
        for sid, s in d["nla"]["stmt"].items():
            st += f"| {NAMES[f]} | NLA-AR | 42 | {sid} | {s['ratio']:.2f} | {s['cosine']:.2f} |\n"
        for L, lay in d["ol"]["layers"].items():
            for sid, s in lay["stmt"].items():
                st += f"| {NAMES[f]} | OL-AR | {L} | {sid} | {s['ratio']:.2f} | {s['cosine']:.2f} |\n"
    calib = {f: d["ol"].get("note") for f, d in res.items()}
    out = "## Reconstruction and concept retention\n\n" + hdr + body + "\n## Statement shifts\n\n" + st
    out += "\n## OL-AR note\n\n```\n" + json.dumps(calib, indent=1) + "\n```\n"
    (HERE / "results" / "summary.md").write_text(out)
    (HERE / "results" / "summary.json").write_text(json.dumps({"rows": rows, "calib": calib}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
