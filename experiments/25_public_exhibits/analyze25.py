"""Table from results/recon25_{row}.json -> results/table25.md (frozen-probe AUROC and class-direction cosine per
rebuild condition, exp 20's layout), then the rescoring of exp 22's scorecard (rescore25.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from lib25 import ROWS, table  # noqa: E402
from rescore25 import load_recon  # noqa: E402
from rescore25 import main as rescore_main  # noqa: E402


def main() -> None:
    recon = load_recon(RES)                       # fails loudly before anything is written
    rows = [recon[r] for r in ROWS]
    md = ["Frozen-probe AUROC against the true layer-42 activations (160 selection items per row):", "", table(rows, "auroc"), "",
          "Cosine of each rebuild's class direction with the true one:", "", table(rows, "class_direction_cos")]
    (RES / "table25.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    rescore_main()


if __name__ == "__main__":
    main()
