"""Tables for RESULTS.md from judged/<view>/<family>/results.json.

    python summarize.py > tables.md
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = HERE / "judged"
FAMS = ["typo_mt", "multilingual_mt", "multihop_mt", "basic_readout_mt"]
# README targets (regex instrument of record 2026-09-23): olens s3d-rl600 / nla-rl-L42 / jlens
TARGETS = {
    "typo_mt": (0.83, 0.75, 0.17),
    "multilingual_mt": (0.62, 0.44, 0.06),
    "multihop_mt": (0.32, 0.22, 0.08),
    "basic_readout_mt": (0.71, 0.53, 0.24),
}
PROMPT_ONLY = {"typo_mt": 0.67, "multilingual_mt": 0.42, "multihop_mt": 0.69,
               "basic_readout_mt": 0.59}
LAYERS = [20, 24, 28, 32, 36, 40, 42, 44, 48, 52, 56, 60]


def res(view: str, fam: str) -> dict | None:
    p = J / view / fam / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(view: str, fam: str) -> str:
    r = res(view, fam)
    if r is None:
        return "n/a"
    n = r["numbers"]
    lo, hi = n["ci95"]
    return f"{n['value']:.2f} [{lo:.2f},{hi:.2f}]"


def inside(view: str, fam: str, target: float) -> str:
    r = res(view, fam)
    if r is None:
        return ""
    lo, hi = r["numbers"]["ci95"]
    return "yes" if lo <= target <= hi else "NO"


def per_layer(view: str, fam: str) -> list[str]:
    r = res(view, fam)
    if r is None:
        return ["n/a"] * len(LAYERS)
    rows = r["rows"]
    return [f"{sum(L in x['passing_layers'] for x in rows) / len(rows):.2f}" for L in LAYERS]


def main() -> None:
    print("## Replication (best-over-11-layers for olens/jlens; L42 for nla)\n")
    print("| family | arm | ours (95% CI) | README | target in CI? | n | prompt-only floor |")
    print("|---|---|---|---|---|---|---|")
    for fam in FAMS:
        for i, (arm, view) in enumerate([("olens", "olens_grid"), ("nla", "nla_L42"),
                                         ("jlens", "jlens_grid")]):
            r = res(view, fam)
            n = r["n_items"] if r else ""
            print(f"| {fam} | {arm} | {cell(view, fam)} | {TARGETS[fam][i]:.2f} | "
                  f"{inside(view, fam, TARGETS[fam][i])} | {n} | {PROMPT_ONLY[fam]:.2f} |")
    print("\n## Single-layer (k=1 sample per cell)\n")
    print("| family | olens L40 | olens L42 | olens L44 | nla L42 | jlens L40 | jlens L42 | jlens L44 |")
    print("|---|---|---|---|---|---|---|---|")
    for fam in FAMS:
        cols = [cell(v, fam) for v in ["olens_L40", "olens_L42", "olens_L44", "nla_L42",
                                       "jlens_L40", "jlens_L42", "jlens_L44"]]
        print(f"| {fam} | " + " | ".join(cols) + " |")
    print("\n## Per-layer pass rate (from the 12-layer judge; item passes AT that layer)\n")
    print("| family | arm | " + " | ".join(f"L{L}" for L in LAYERS) + " |")
    print("|---|---|" + "---|" * len(LAYERS))
    for fam in FAMS:
        for arm in ["olens", "jlens"]:
            print(f"| {fam} | {arm} | " + " | ".join(per_layer(f"{arm}_all", fam)) + " |")


if __name__ == "__main__":
    main()
