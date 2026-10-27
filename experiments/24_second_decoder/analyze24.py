"""Tables from results/recon24_{row}.json -> results/table24.md (layer 44, then a layer-40 block beneath) and
results/table24.json (the per-row scores, both layers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib24 import LAYERS, table  # noqa: E402

ROW_ORDER = ("py_string", "js_string", "latex_math", "py_arg", "json_depth", "html_depth", "md_heading", "md_list",
             "tb_tablecol", "tb_paren")


def main() -> None:
    res = HERE / "results"
    rows = [json.loads(p.read_text()) for p in sorted(res.glob("recon24_*.json"))]
    rows.sort(key=lambda r: ROW_ORDER.index(r["feature"]) if r["feature"] in ROW_ORDER else len(ROW_ORDER))
    if not rows:
        print("no results/recon24_*.json yet")
        return
    variants = sorted({r.get("variant") for r in rows})
    md = [f"# Exp 24 tables ({len(rows)} rows, {rows[0]['n']} selection items per row, OL-AR variant {', '.join(map(str, variants))})", ""]
    for L in LAYERS:
        md += [f"## OL-AR layer {L}" + (" (primary)" if L == LAYERS[0] else " (check)"), "", table(rows, layer=str(L))]
    text = "\n".join(md)
    (res / "table24.md").write_text(text)
    (res / "table24.json").write_text(json.dumps({r["feature"]: {"n": r["n"], "variant": r.get("variant"), "layers": r["layers"]}
                                                  for r in rows}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
