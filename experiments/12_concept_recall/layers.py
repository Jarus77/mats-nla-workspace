"""Neel's second table: OLens at layers 40 / 42 / 44 as three 'methods'. Prompted AUROC per layer
(exp 07 frozen templates, mean of selected) and judged spontaneous stated / exhibited rates."""
from __future__ import annotations

import json
from pathlib import Path

from analyze import NAMES
from combine import ORDER
from recall_lib import aggregate

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / "experiments" / "07_template_mining" / "templates_frozen.json"
LAYERS = ("40", "42", "44")


def prompted_by_layer() -> dict[str, dict[str, float]]:
    t = json.loads(FROZEN.read_text())["templates"]["ol"]
    return {f: {L: t[f]["mean_of_selected"][f"ol@{L}"]["auroc"] for L in LAYERS} for f in ORDER}


def spontaneous_by_layer() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for f in ORDER:
        for L in LAYERS:
            p = HERE / "results" / "judged" / f"{f}_ol{L}.json"
            if p.exists():
                d = json.loads(p.read_text())
                out.setdefault(f, {})[L] = aggregate(d["judged"], d["values"])
    return out


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def main() -> None:
    pr, sp = prompted_by_layer(), spontaneous_by_layer()
    lines = ["| concept type | prompted 40 / 42 / 44 | states it 40 / 42 / 44 | exhibits it 40 / 42 / 44 |",
             "|---|---|---|---|"]
    rows = []
    for f in ORDER:
        s = sp.get(f, {})
        st = [s.get(L, {}).get("stated_rate") for L in LAYERS]
        ex = [s.get(L, {}).get("exhibited_correct_rate") for L in LAYERS]
        rows.append({"feature": f, "prompted": pr[f], "stated": dict(zip(LAYERS, st)),
                     "exhibited": dict(zip(LAYERS, ex))})
        lines.append(f"| {NAMES[f]} | " + " / ".join(f"{pr[f][L]:.2f}" for L in LAYERS) + " | "
                     + " / ".join(_pct(x) for x in st) + " | " + " / ".join(_pct(x) for x in ex) + " |")
    md = ("# OLens by input layer (40 / 42 / 44), same 14 concept types\n\nPrompted = AUROC of the mean of the "
          "three frozen OLens templates at that layer (160 items). States / exhibits = judged free readouts "
          "(40 items x 8 samples).\n\n" + "\n".join(lines) + "\n")
    (HERE / "results" / "ol_by_layer.md").write_text(md)
    (HERE / "results" / "ol_by_layer.json").write_text(json.dumps(rows, indent=1))
    print(md)


if __name__ == "__main__":
    main()
