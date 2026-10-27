"""Table from results/recon20_{row}.json -> results/table20.md (frozen-probe AUROC per rebuild condition)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib20 import table  # noqa: E402


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("recon20_*.json"))]
    md = table(rows)
    (HERE / "results" / "table20.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
