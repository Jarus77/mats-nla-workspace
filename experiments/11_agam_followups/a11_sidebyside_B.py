"""A4 continued: side-by-sides from the Phase B job (Oracle text for the planted and operand vectors, the col
P2 cell at k=8, J-Lens top-10 lists for the same vectors), plus two sanity numbers (OL names the planted
country; OL quotes the operands).

    ../../.venv/bin/python a11_sidebyside_B.py   -> results/sidebyside_B.md, results/sidebyside_B.json
PILOT / EXPLORATORY.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
PB = HERE / "results" / "phaseB"
OUT = HERE / "results"
SEED = 11
N_SHOW = 6


def clip(t: str | None, n: int = 380) -> str:
    if not t:
        return "(none)"
    t = t.replace("\n", " ⏎ ").strip()
    return t if len(t) <= n else t[:n] + " …"


def q(t: str) -> str:
    return "> " + t


def main() -> None:
    rng = random.Random(SEED)
    g = torch.load(PB / "ol_gens.pt", weights_only=False)
    top = torch.load(PB / "jl_top10.pt", weights_only=False)
    rb = torch.load(EXP / "05_planted_intermediates" / "results" / "readers_base.pt", weights_only=False)
    names = rb["names"]  # per item: (planted Y, original X)
    rj = json.loads((EXP / "05_planted_intermediates" / "results" / "readers.json").read_text())
    nla_ex = {tuple(e["names"]): e for e in rj["nla_examples"]}
    ar = [json.loads(l) for l in (EXP / "10_operand_gate" / "data" / "arithmetic.jsonl").read_text().splitlines()]
    items = json.loads((HERE / "data" / "items.json").read_text())["col"]
    nla_gens = torch.load(PB / "nla_gens.pt", weights_only=False)["col_k8@42"] if (PB / "nla_gens.pt").exists() else None
    stats: dict = {}

    # ---- planted bridge: OL names the planted country? (string match on the country name, any sample)
    def names_in(texts: list[str], name: str) -> bool:
        return any(name.lower() in (t or "").lower() for t in texts)
    for L in (42, 44):
        for cond in ("natural", "planted"):
            hitY = sum(names_in(g[f"planted@{L}"][cond][i], names[i][0]) for i in range(200)) / 200
            hitX = sum(names_in(g[f"planted@{L}"][cond][i], names[i][1]) for i in range(200)) / 200
            stats[f"planted@{L}|{cond}"] = {"any_sample_names_planted_Y": hitY, "any_sample_names_original_X": hitX}
    jl_hitY = {cond: sum(any(names[i][0].lower() in t.lower() for t, _ in top[f"planted_{cond}"][i]) for i in range(200)) / 200 for cond in ("natural", "planted")}
    stats["jlens_top10_names_planted_Y"] = jl_hitY
    # ---- operands: does OL quote the prompt's expression? (first "a op b" in any sample)
    expr = re.compile(r"(\d+)\s*([+\-−])\s*(\d+)")
    ok = tot = 0
    for i, row in enumerate(g["operand@42"]):
        it = ar[i]
        quoted = [m for t in row for m in expr.findall(t or "")]
        if quoted:
            tot += 1
            ok += any((int(a), int(b)) == (it["a"], it["b"]) for a, _, b in quoted)
    stats["operand@42"] = {"n": len(g["operand@42"]), "samples_with_expression": tot, "prompt_expression_quoted_correctly": ok / max(1, tot),
                           "jlens_top10_contains_a_digit": sum(any(t.strip().isdigit() for t, _ in top["operand"][i]) for i in range(len(g["operand@42"]))) / len(g["operand@42"])}

    md = ["# Side-by-side readouts from the Phase B job (A4, continued) — PILOT / EXPLORATORY\n",
          f"Sanity numbers: {json.dumps(stats, indent=1)}\n",
          "## Planted bridge (exp 05 vectors): NLA (stored) · Oracle Lens k=4 at 42 and 44 · J-Lens top-10\n"]
    for i in rng.sample(range(200), N_SHOW):
        Y, X = names[i]
        ex = nla_ex.get((Y, X))
        md += [f"### item {i}: planted {Y} (original {X})",
               f"**J-Lens@42 top-10, natural → planted:** {[t for t, _ in top['planted_natural'][i]]} → {[t for t, _ in top['planted_planted'][i]]}  ",
               f"**OL@42 natural:** {clip(g['planted@42']['natural'][i][0])}  ",
               f"**OL@42 planted:** {clip(g['planted@42']['planted'][i][0])}  ",
               f"**OL@44 planted:** {clip(g['planted@44']['planted'][i][0], 220)}  "]
        if ex:
            md += [f"**NLA@42 natural:** {clip(ex['natural'][0], 300)}  ", f"**NLA@42 planted:** {clip(ex['planted'][0], 300)}  "]
        md.append("")
    md += ["## Arithmetic operands (exp 10 vectors, bare prompt, '=' at layer 42): NLA (Kenney's) · OL k=4 · J-Lens top-10\n",
           "| prompt | OL@42 (2 of 4 samples) | J-Lens top-10 | NLA (Kenney, sample 1) |", "|---|---|---|---|"]
    for i in rng.sample(range(len(g["operand@42"])), N_SHOW):
        it = ar[i]
        md.append(f"| `{it['text']}` | {clip(g['operand@42'][i][0], 160)} ⫽ {clip(g['operand@42'][i][1], 160)} | {[t for t, _ in top['operand'][i]]} | {clip(it['nla_readouts'][0], 200)} |")
    md += ["", "## col P2 cell (exp 08 items, layer 42): zero vs ± J-removed 4r* q90; OL k=8 at 42 and 44, NLA k=8 at 42, J-Lens top-10\n"]
    for i in rng.sample(range(60), N_SHOW):
        it = items[i]
        md += [f"### {it['id']} (col {it['col']}, chars remaining {it['chars_remaining']})",
               f"**text tail:** {clip(it['text'][-200:], 200)}  ",
               f"**J-Lens@42 top-10 zero / +Δ / −Δ:** {[t for t, _ in top['col_zero'][i]]} / {[t for t, _ in top['col_jrem4+'][i]]} / {[t for t, _ in top['col_jrem4-'][i]]}  "]
        for cond in ("zero", "jrem4+", "jrem4-"):
            md.append(f"**OL@42 {cond}:**  ")
            md += [q(clip(t, 240)) for t in g["col_k8@42"][cond][i][:2]]
            if nla_gens:
                md.append(f"**NLA@42 {cond}:**  ")
                md += [q(clip(t, 240)) for t in nla_gens[cond][i][:2]]
        md.append("")
    (OUT / "sidebyside_B.md").write_text("\n".join(md))
    (OUT / "sidebyside_B.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
