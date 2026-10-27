"""Exp 27 bank (laptop): exp 23's builder at 400 random read points per kind (py, js, json, tex, md; 2,000 in all),
seed 2727, the same public document pool (package directories on this machine plus arXiv e-prints) and the same
state labelling (lib23.labels). Imports build23 and runs its main() with this directory, seed and count in place
of exp 23's, so the sampling loop is exp 23's code and not a copy. Writes results/items_base.json (tracked) and
results/build_info.json (sha256, seed, counts per state, document kinds). Then:

    modal volume put rfm-results results/items_base.json 27_random_2000/items_base.json

    .venv/bin/python build27.py --arxiv <dir with src_*/ e-print folders> [--home ~]

Note: exp 22's arXiv pool held 339 .tex files over 4 KB and the builder takes one point per file, so the tex kind
caps at the size of the pool; the other kinds have thousands of files each. build_info.json records what was found.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "23_random_points"))
import build23 as B23  # noqa: E402  (also puts exp 22 and exp 23 on sys.path)

SEED = 2727
N_PER_KIND = 400                       # x 5 kinds = 2,000 points

# build23.main() reads these module globals at call time (output directory, seed, count); everything else (the
# document pool, candidate points, state labels, tokeniser, truncation, build_info fields) is unchanged.
SETTINGS = {"HERE": HERE, "SEED": SEED, "N_PER_KIND": N_PER_KIND}


def main() -> None:
    for k, v in SETTINGS.items():
        setattr(B23, k, v)
    (HERE / "results").mkdir(exist_ok=True)
    B23.main()


if __name__ == "__main__":
    main()
