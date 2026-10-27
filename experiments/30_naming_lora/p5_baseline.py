"""P5 baseline (design (c) of the memo): within-item FVE of naming vs non-naming samples on exp 22's existing k = 8
NLA readouts (`gen_nla_{row}.json`). Those files carry raw texts only (no per-sample FVE: exp 22 never ran the
reconstructor on them), in which case this script records P5 as skipped. If a file `results/p5_fve.json` of the form
{row: {item id: [[names_state, fve], ...]}} is ever produced (one H200 pass of the AR over the 2,560 readouts), it is
scored here with lib30.within_item_fve_diff."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from lib30 import TRAIN_ROWS, within_item_fve_diff  # noqa: E402

GEN = HERE.parent / "22_public_bank" / "results"


def main() -> dict:
    RES.mkdir(exist_ok=True)
    scored = RES / "p5_fve.json"
    if scored.exists():
        by_row = json.loads(scored.read_text())
        rows = {row: within_item_fve_diff({i: [(bool(a), float(b)) for a, b in s] for i, s in by_row[row].items()}) for row in by_row}
        pooled = within_item_fve_diff({f"{row}:{i}": [(bool(a), float(b)) for a, b in s] for row in by_row for i, s in by_row[row].items()})
        out = {"skipped": False, "rows": rows, "pooled": pooled}
    else:
        files = {row: GEN / f"gen_nla_{row}.json" for row in TRAIN_ROWS}
        have_fve = {row: ("fve" in json.loads(p.read_text())) if p.exists() else None for row, p in files.items()}
        out = {"skipped": True, "reason": "exp 22's gen_nla_{row}.json carry k = 8 raw texts and no per-sample FVE; the reconstructor "
                                          "was never run on them, so the within-item comparison cannot be scored from existing files",
               "gen_nla_has_fve": have_fve}
    (RES / "p5.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
