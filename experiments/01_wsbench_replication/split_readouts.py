"""Split the 12-layer produced readouts (GRID 20..60 step 4 + 42) into judge-ready arms.

    python split_readouts.py            # readouts/<arm>/<fam>.jsonl -> judge_in/<arm>_<view>/<fam>.jsonl

Views: grid (the 11 README layers, best-over-layers like the published numbers) and one file per
single layer (L40, L42, L44). The MT judges read the layer set from the file, so a single-layer
file is judged as-is (no `layers=` subset flag). nla is L42 only.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "readouts", HERE / "judge_in"
GRID = set(range(20, 61, 4))
VIEWS = {"grid": GRID, "L40": {40}, "L42": {42}, "L44": {44}}


def split(arm_dir: Path) -> list[str]:
    made = []
    arm = arm_dir.name
    views = {"L42": {42}} if arm == "nla" else VIEWS
    for f in sorted(arm_dir.glob("*.jsonl")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        for view, layers in views.items():
            keep = [r for r in rows if r["layer"] in layers]
            if not keep:
                continue
            out = DST / f"{arm}_{view}" / f.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                           encoding="utf-8")
            made.append(f"{out.relative_to(HERE)} ({len(keep)} rows)")
    return made


if __name__ == "__main__":
    for d in sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.endswith("_smoke")):
        print("\n".join(split(d)))
